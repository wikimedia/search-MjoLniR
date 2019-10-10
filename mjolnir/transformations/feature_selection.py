"""Reduce the count of features in a feature set"""

from typing import cast, Dict, List

from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame, functions as F

import mjolnir.feature_engineering
import mjolnir.transform as mt


def explode_features(metadata: Dict) -> mt.Transformer:
    def transform(df: DataFrame) -> DataFrame:
        # We drop the features column when exploding, so we need to hold
        # onto the metadata somewhere else.
        metadata['input_feature_meta'] = df.schema['features'].metadata
        # While later code could guess what columns of the exploded dataframe were
        # in the input, based on the feature metadata, be more explicit and
        # keep track directly.
        metadata['default_cols'] = [x for x in df.columns if x != 'features']
        return mjolnir.feature_engineering.explode_features(df) \
            .drop('features')
    return transform


def select_features(
    wiki: str,
    num_features: int,
    metadata: Dict
) -> mt.Transformer:
    def transform(df: DataFrame) -> DataFrame:
        # Compute the "best" features, per some metric
        sc = df.sql_ctx.sparkSession.sparkContext
        features = metadata['input_feature_meta']['features']
        selected = mjolnir.feature_engineering.select_features(
            sc, df, features, num_features, algo='mrmr')
        metadata['wiki_features'][wiki] = selected

        # Rebuild the `features` col with only the selected features
        keep_cols = metadata['default_cols'] + selected
        df_selected = df.select(*keep_cols)
        assembler = VectorAssembler(
            inputCols=selected, outputCol='features')
        return assembler.transform(df_selected).drop(*selected)
    return transform


def attach_feature_metadata(metadata: Dict) -> mt.Transformer:
    def transform(df: DataFrame) -> DataFrame:
        feature_meta = dict(
            metadata['input_feature_meta'],
            wiki_features=metadata['wiki_features'])
        sc = df.sql_ctx.sparkSession.sparkContext
        return df.withColumn(
            'features',
            mjolnir.spark.add_meta(sc, F.col('features'), feature_meta))
    return transform


@mt.typed_transformer(mt.FeatureVectors, mt.FeatureVectors, __name__)
def transformer(
    df_label: DataFrame,
    temp_dir: str,
    wikis: List[str],
    num_features: int
) -> mt.Transformer:
    mt.check_schema(df_label, mt.LabeledQueryPage)

    # Hack to transfer metadata between transformations. This is populated in
    # time since `select_features` does direct computation of the features.
    metadata = cast(Dict, {'wiki_features': {}})

    return mt.seq_transform([
        mt.restrict_wikis(wikis),
        mt.join_labels(df_label),
        explode_features(metadata),
        mt.cache_to_disk(temp_dir, partition_by='wikiid'),
        mt.for_each_item('wikiid', wikis, lambda wiki: select_features(
            wiki, num_features, metadata)),
        attach_feature_metadata(metadata),
        # While we used the labels for selecting features, they are not part of the feature vectors.
        # Allow them to be joined with any other label set for export to training.
        lambda df: df.drop('cluster_id', 'label'),
        lambda df: df.repartition(200, 'wikiid', 'query'),
    ])
