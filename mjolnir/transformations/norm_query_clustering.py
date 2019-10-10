"""Cluster queries based on a normalized form and query result similarity"""

from pyspark.sql import DataFrame, functions as F, types as T, Window

import mjolnir.kafka
from mjolnir.kafka.client import ClientConfig
from mjolnir.norm_query import _make_query_groups
from mjolnir.spark import at_least_n_distinct
import mjolnir.transform as mt


def with_norm_query(df: DataFrame) -> DataFrame:
    return df.withColumn(
        'norm_query',
        F.expr('stemmer(query, substring(wikiid, 1, 2))'))


def filter_min_sessions_per_norm_query(min_sessions: int) -> mt.Transformer:
    def transform(df: DataFrame) -> DataFrame:
        w = Window.partitionBy('wikiid', 'norm_query')
        return (
            df.withColumn(
                'has_min_sessions',
                at_least_n_distinct('session_id', min_sessions).over(w))
            .where(F.col('has_min_sessions'))
            .drop('has_min_sessions'))
    return transform


def as_unique_queries(df: DataFrame) -> DataFrame:
    return (
        df
        .select('wikiid', 'norm_query', 'query')
        .drop_duplicates())


def with_hit_page_ids(brokers: ClientConfig, top_n: int) -> mt.Transformer:
    def transform(df: DataFrame) -> DataFrame:
        return mjolnir.es_hits.transform(
            df, brokers=brokers, top_n=top_n,
            indices=mt.ContentIndices())
    return transform


def cluster_within_norm_query_groups(df: DataFrame) -> DataFrame:
    make_groups = F.udf(_make_query_groups, T.ArrayType(T.StructType([
        T.StructField('query', T.StringType(), nullable=False),
        T.StructField('norm_query_group_id', T.IntegerType(), nullable=False),
    ])))
    return (
        df
        .groupBy('wikiid', 'norm_query')
        .agg(F.collect_list(F.struct('query', 'hit_page_ids')).alias('source'))
        .select(
            'wikiid', 'norm_query',
            F.explode(make_groups('source')).alias('group'))
        .select('wikiid', 'norm_query', 'group.query', 'group.norm_query_group_id'))


def with_unique_cluster_id(df: DataFrame) -> DataFrame:
    return (
        df
        .groupby('wikiid', 'norm_query', 'norm_query_group_id')
        .agg(F.collect_list('query').alias('queries'))
        .select(
            'wikiid', 'queries',
            F.monotonically_increasing_id().alias('cluster_id'))
        .select('wikiid', F.explode('queries').alias('query'), 'cluster_id'))


@mt.typed_transformer(mt.QueryClicks, mt.QueryClustering, __name__)
def transformer(
    brokers: str, topic_request: str, topic_response: str,
    top_n: int, min_sessions_per_query: int
) -> mt.Transformer:
    kafka_config = ClientConfig(
        brokers, topic_request, topic_response,
        mjolnir.kafka.TOPIC_COMPLETE)
    return mt.seq_transform([
        with_norm_query,
        filter_min_sessions_per_norm_query(min_sessions_per_query),
        as_unique_queries,
        with_hit_page_ids(kafka_config, top_n),
        cluster_within_norm_query_groups,
        with_unique_cluster_id,
        lambda df: df.repartition(200, 'wikiid', 'query')
    ])
