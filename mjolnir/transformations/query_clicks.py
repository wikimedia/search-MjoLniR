"""Consolidate external logs into a single input datasource

Takes external click logs, such as `discovery.query_clicks_daily`,
and consolidates them into a single consistent partition of input
data to be used by later transformations. Data is partitioned
by a `snapshot_id` column which flows through into all later
tables built from this data source.
"""

from pyspark.sql import DataFrame, functions as F

import mjolnir.transform as mt


def filter_high_volume_ip(max_q_by_day: int) -> mt.Transformer:
    def transform(df: DataFrame) -> DataFrame:
        return (
            df
            .where(F.col('q_by_ip_day') < max_q_by_day)
            .drop('q_by_ip_day'))
    return transform


def with_page_ids(df: DataFrame) -> DataFrame:
    return (
        df
        .withColumn('hit_page_ids', F.col('hits.pageid'))
        .drop('hits')
        .withColumn('click_page_ids', F.col('clicks.pageid'))
        .drop('clicks'))


# TODO: Schema validation needs to be enhanced to handle sub-fields like
# hits.pageid better before it's schema can be specified.
@mt.typed_transformer(None, mt.QueryClicks, __name__)
def transformer(max_q_by_day: int) -> mt.Transformer:
    return mt.seq_transform([
        filter_high_volume_ip(max_q_by_day),
        with_page_ids,
        lambda df: df.repartition(200, 'wikiid', 'query')
    ])
