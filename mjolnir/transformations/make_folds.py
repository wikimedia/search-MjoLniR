"""Convert labeled vectors into folded training inputs

Currently converts to svmrank and xgboost formats.
"""
import multiprocessing.dummy
import os
from typing import Any, List, Optional, TypeVar

from pyspark.sql import DataFrame, functions as F, Row, types as T

from mjolnir.training.tuning import group_k_fold
import mjolnir.transform as mt
from mjolnir.utils import as_local_path, as_output_file


_T = TypeVar('_T')


def frac_label_to_int(col_name, scale=10) -> mt.Transformer:
    """Convert a label from fractional to integer.

    Maps linearly from the domain [0-1] to [0-scale]. Does
    not verify input data is in the expected domain.
    """
    col = F.round(F.col(col_name) * scale).cast('int')

    def transform(df: DataFrame) -> DataFrame:
        return df.withColumn(col_name, col)
    return transform


def attach_fold_col(num_folds: int, fold_col: str) -> mt.Transformer:
    def transform(df: DataFrame) -> DataFrame:
        # TODO: rename upstream
        df = df.withColumnRenamed('cluster_id', 'norm_query_id')
        df = group_k_fold(df, num_folds, fold_col)
        # cluster_id was only needed to determine which records must be
        # in the same folds.
        # TODO: This assumption is hardcoded from the dbn labeling algorithm,
        # can it be generalized (upstream probably)?
        return df.drop('norm_query_id')
    return transform


def partition_and_order_for_output(df: DataFrame) -> DataFrame:
    # Put data in the correct order for being written out to files:
    # all results for a query grouped together on the same partition.
    # TODO: Why ascending label order? Downstream requirement?
    return (
        df
        .repartition(200, 'wikiid', 'query')
        .sortWithinPartitions('wikiid', 'query', F.col('label').asc())
    )


def simplify_datawriter_paths(j_paths: Any) -> List[Row]:
    rows = []
    for fold_id, j_fold in enumerate(j_paths):
        for split_name, file_path in dict(j_fold).items():
            rows.append(Row(
                vec_format='svmrank',
                split_name=split_name,
                path=file_path,
                fold_id=fold_id))
    return rows


# wikiid doesn't exist at this level, the dataframes already
# represent a single wiki and that is added back in later.
TrainingFilesNoWiki = T.StructType([
    field for field in mt.TrainingFiles if field.name != "wikiid"])


def convert_mllib_to_svmrank(
    path_format: str, fold_col: Optional[str], num_folds: int
) -> mt.Transformer:
    if fold_col is None and num_folds != 1:
        raise Exception('num_folds must be 1 when fold_col is None, got: {}'.format(num_folds))

    def transform(df: DataFrame) -> DataFrame:
        sc = df.sql_ctx.sparkSession.sparkContext

        jsc, jvm = sc._jsc, sc._jvm  # type: ignore
        writer = jvm.org.wikimedia.search.mjolnir.DataWriter(jsc, False)

        jdf = df._jdf  # type: ignore
        j_paths = writer.write(jdf, path_format, fold_col, num_folds)
        # We use this flattened shape so we can distribute the rows for
        # conversion to xgboost, one file per partition.
        # TODO: Should DataWriter return this shape
        # directly? We need to re-construct this later as it's the shape
        # training expects anyways.
        training_files = list(simplify_datawriter_paths(j_paths))

        # Not the most efficient thing to store this tiny array as a dataframe,
        # but we need to process it's contents on executors to convert to other
        # formats, or do training and it keeps with out Transformer interface.
        return df.sql_ctx.sparkSession.createDataFrame(
            training_files, TrainingFilesNoWiki)  # type: ignore
    return transform


def _convert_xgboost_local(in_path: str, out_path: str) -> None:
    import xgboost
    # TODO: Do feature names save? should we set them?
    xgboost.DMatrix(in_path).save_binary(out_path)


def _convert_xgboost_remote(in_path: str, out_path: str) -> None:
    with as_local_path(in_path) as local_input, \
            as_output_file(out_path, 'wb', overwrite=True) as local_output:
        _convert_xgboost_local(local_input, local_output.name)


def convert_svmrank_to_xgboost(df: DataFrame) -> DataFrame:
    def convert_one(row: Row) -> Row:
        # For now place the .xgb right next to the svmrank files. Naming/path
        # options could be added if needed later.
        out_path = row.path + '.xgb'
        _convert_xgboost_remote(row.path, out_path)
        return Row(**dict(
            row.asDict(),
            vec_format='xgboost',
            path=out_path))

    # Each row represents potentially gigabytes, convince spark
    # to create a partition per row.
    rdd_xgb = mt.partition_per_row(df.rdd).map(convert_one)
    df_xgb = df.sql_ctx.createDataFrame(rdd_xgb, df.schema)  # type: ignore
    # Return both the xgb and svmrank datasets since
    # we aren't purging the related files. df is safe to reuse since
    # svmrank conversion returns a new dataframe with no lineage.
    return df.union(df_xgb)


def convert_mllib_to_svmrank_and_xgboost(
    path_format: str,
    fold_col: Optional[str],
    num_folds: int
) -> mt.Transformer:
    return mt.seq_transform([
        convert_mllib_to_svmrank(path_format, fold_col, num_folds),
        convert_svmrank_to_xgboost,
    ])


@mt.typed_transformer(mt.FeatureVectors, mt.TrainingFiles, __name__)
def transformer(
    df_label: DataFrame, wiki: str, training_output_path: str, num_folds: int
) -> mt.Transformer:
    mt.check_schema(df_label, mt.LabeledQueryPage)

    fold_col = 'fold'
    # Format for individual training files. First %s is split name, second
    # is the fold id (a 0 indexed number, or `x` for un-folded)
    path_format = os.path.join(training_output_path, wiki + '.%s.f%s')

    # This pool should be sized to run all possible tasks, currently two
    # (folded and unfolded). We shouldn't be limiting any concurrency here.
    task_pool = multiprocessing.dummy.Pool(2)
    restrict_wiki = mt.restrict_wikis([wiki])

    return mt.seq_transform([
        restrict_wiki,
        mt.assert_not_empty,
        mt.join_labels(restrict_wiki(df_label)),
        # TODO: hardcoded assumption about DBN, labels
        # could be on a variety of scales. Maybe metadata could
        # be attached to label col to inform this?
        frac_label_to_int('label'),
        # hardcoded assumption that labels with same cluster_id
        # are not independent and must be in the same fold.
        attach_fold_col(num_folds, fold_col),
        partition_and_order_for_output,
        # This may recalculate the above per output, but folds were
        # calculated on the driver ensuring those will stay constant.
        # Everything else is preferrable to recalculate rather than
        # having many executors cache it in memory while 1 executor
        # spends 20 minutes writing out datasets.
        mt.par_transform([
            # Write out folded dataset
            convert_mllib_to_svmrank_and_xgboost(
                path_format, fold_col, num_folds),
            # Write out unfolded "all" dataset. The resulting rows
            # are distinguished from above with `split_name` of all.
            convert_mllib_to_svmrank_and_xgboost(
                path_format, fold_col=None, num_folds=1),
        ], mapper=task_pool.imap_unordered),
        lambda df: df.withColumn('wikiid', F.lit(wiki)),
        # After the above we have a row per training file, with most
        # data represented externally via hdfs paths. No need for
        # multiple partitions.
        lambda df: df.repartition(1),
    ])
