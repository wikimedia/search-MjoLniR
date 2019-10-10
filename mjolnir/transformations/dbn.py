"""Learn relevance labels from click logs and a clustering of queries

"""
from typing import Mapping, Union

from pyspark.sql import DataFrame, functions as F

import mjolnir.dbn
import mjolnir.transform as mt


def with_exploded_hits(df: DataFrame) -> DataFrame:
    return (
        df
        .select(
            F.posexplode('hit_page_ids').alias('hit_position', 'hit_page_id'),
            *df.columns)
        .drop('hit_page_ids')
        .withColumn('clicked', F.expr('array_contains(click_page_ids, hit_page_id)'))
        .drop('click_page_ids'))


def as_labeled_clusters(
    dbn_config: Mapping[str, Union[int, float]]
) -> mt.Transformer:
    def transform(df: DataFrame) -> DataFrame:
        return mjolnir.dbn.train(df, dbn_config) \
            .withColumnRenamed('relevance', 'label')
    return transform


@mt.typed_transformer(mt.QueryClicks, mt.LabeledQueryPage, __name__)
def transformer(df_cluster: DataFrame) -> mt.Transformer:
    mt.check_schema(df_cluster, mt.QueryClustering)
    return mt.seq_transform([
        # Attach cluster id's to search queries
        # TODO: Stats about # of queries that didn't have a cluster id and were discarded
        mt.join_cluster_by_query(df_cluster),
        # Massage into dbn expected format
        with_exploded_hits,
        # Run the labeling process
        mt.temp_rename_col(
            'cluster_id', 'norm_query_id',
            as_labeled_clusters({
                'MIN_DOCS_PER_QUERY': 10,
                'MAX_DOCS_PER_QUERY': 20,
                'DEFAULT_REL': 0.5,
                'MAX_ITERATIONS': 40,
                'GAMMA': 0.9,
            })),
        # labeling gave results per cluster_id, transform into results per
        # query.  This ends up labeling all queries in df_cluster, rather than
        # the dataframe being transformed. For standard usage where the
        # dataframe being transformed is the same as the dataframe that was
        # input to clustering this is acceptable, but if the join filters a lot
        # of things out the results might be unexpected.
        mt.join_cluster_by_cluster_id(df_cluster),
        # Rename things to match our output tables
        # TODO: Rename upstream in mjolnir
        lambda df: df.select(
            'wikiid', 'query', F.col('hit_page_id').alias('page_id'),
            'label', 'cluster_id'),
        # TODO: Any interesting metadata to attach to `label` column? The config passed
        # to as_labeled_clusters?
        lambda df: df.repartition(200, 'wikiid', 'query')
    ])
