"""Collect feature vectors from elasticsearch"""

import datetime
from typing import List, Mapping

from pyspark import Accumulator
from pyspark.sql import DataFrame, functions as F

import mjolnir.features
import mjolnir.kafka
from mjolnir.kafka.client import ClientConfig
import mjolnir.spark
import mjolnir.transform as mt


def _check_features(fnames_accu: Accumulator) -> List[str]:
    num_rows_collected = set(fnames_accu.value.values())
    if len(num_rows_collected) != 1:
        # Something went sideways, we might have a bad dataset. Safer to try
        # again later.
        raise Exception('Did not collect equal number of rows per feature')
    return list(fnames_accu.value.keys())


def _add_meta(df: DataFrame, col_name: str, metadata: Mapping) -> DataFrame:
    sc = df.sql_ctx.sparkSession.sparkContext
    return df.withColumn(
        col_name,
        mjolnir.spark.add_meta(sc, F.col(col_name), metadata))


def collect_features(
    kafka_config: ClientConfig, feature_set: str
) -> mt.Transformer:
    def transform(df: DataFrame) -> DataFrame:
        df_features, fnames_accu = mjolnir.features.collect(
            df,
            model='featureset:' + feature_set,
            brokers=kafka_config,
            indices=mt.ContentIndices())
        # Collect the accumulator to get feature names
        df_features.cache().count()
        # Future transformations have to be extra careful to not lose this metadata
        return _add_meta(df_features, 'features', {
            'feature_set': feature_set,
            'features': _check_features(fnames_accu),
            'collected_at': datetime.datetime.now().isoformat()
        })
    return transform


@mt.typed_transformer(mt.QueryPage, mt.FeatureVectors, __name__)
def transformer(
    brokers: str, topic_request: str, topic_response: str,
    feature_set: str,
) -> mt.Transformer:
    """Collect feature vectors for the provided query/page pairs

    Parameters
    ----------
    brokers :
        Comma separated list of kafka hosts to bootstrap from.
    topic_request :
        Kafka topic to send feature vector requests on.
    topic_response :
        Kafka topic to recieve feature vector responses on.
    feature_set :
        A named elasticsearch ltr featureset to collect features from.

    Returns
    -------
    A Transformer accepting mt.QueryPage and returning mt.FeatureVectors.
    """
    kafka_config = ClientConfig(
        brokers, topic_request, topic_response,
        mjolnir.kafka.TOPIC_COMPLETE)
    return mt.seq_transform([
        # TODO: Rename cols upstream in mjolnir
        mt.temp_rename_col(
            'page_id', 'hit_page_id',
            collect_features(kafka_config, feature_set)),
        lambda df: df.select('wikiid', 'query', 'page_id', 'features'),
        # TODO: Should write_partition also correctly partition all
        # our datasets by (wikiid, query)? This would help joins against
        # the data not require a shuffle.
        lambda df: df.repartition(200, 'wikiid', 'query')
    ])
