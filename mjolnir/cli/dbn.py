from argparse import ArgumentParser
from typing import Callable

from mjolnir.cli.helpers import Cli, HivePartition
from mjolnir.transformations import dbn

from pyspark.sql import DataFrame


def transform(
    query_clicks: HivePartition,
    query_clustering: HivePartition,
    **kwargs
) -> DataFrame:
    transformer = dbn.transformer(query_clustering.df)
    return transformer(query_clicks.df)


def configure(parser: ArgumentParser) -> Callable:
    main = Cli('dbn', transform, parser)
    # I/O
    main.require_query_clicks_partition()
    main.require_query_clustering_partition()
    main.require_output_table([('algorithm', 'dbn')])
    return main
