from argparse import ArgumentParser
from typing import Callable

from mjolnir.cli.helpers import Cli
from mjolnir.transformations import query_clicks

from pyspark.sql import DataFrame


def transform(df_in: DataFrame, max_q_by_day: int, **kwargs) -> DataFrame:
    transformer = query_clicks.transformer(max_q_by_day)
    return transformer(df_in)


def configure(parser: ArgumentParser) -> Callable:
    main = Cli('query_clicks_ltr', transform, parser)
    # I/O
    main.require_daily_input_table()
    main.require_output_table([])
    # script parameters
    main.add_argument('--max-q-by-day', type=int, default=50)
    return main
