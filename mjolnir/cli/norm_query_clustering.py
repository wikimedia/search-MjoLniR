from argparse import ArgumentParser
from typing import Callable

from mjolnir.cli.helpers import Cli, HivePartition
from mjolnir.transformations import norm_query_clustering

from pyspark.sql import DataFrame, SparkSession


def register_stemmer(spark: SparkSession) -> None:
    spark.sql("DROP TEMPORARY FUNCTION IF EXISTS stemmer")
    spark.sql("CREATE TEMPORARY FUNCTION stemmer AS "
              "'org.wikimedia.analytics.refinery.hive.StemmerUDF'")


def transform(
    spark: SparkSession,
    query_clicks: HivePartition,
    brokers: str,
    topic_request: str,
    topic_response: str,
    top_n: int,
    min_sessions_per_query: int,
    **kwargs
) -> DataFrame:
    register_stemmer(spark)
    transformer = norm_query_clustering.transformer(
        brokers, topic_request, topic_response,
        top_n, min_sessions_per_query)
    return transformer(query_clicks.df)


def configure(parser: ArgumentParser) -> Callable:
    main = Cli('norm_query_clustering', transform, parser)
    # I/O
    main.require_kafka_arguments()
    main.require_query_clicks_partition()
    main.require_output_table([('algorithm', 'norm_query')])

    # script parameterization
    main.add_argument('--top-n', type=int, default=10)
    main.add_argument('--min-sessions-per-query', type=int, default=10)
    return main
