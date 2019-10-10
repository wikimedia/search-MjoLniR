"""
Tests for mjolnir.spark.*
"""

from __future__ import absolute_import
import mjolnir.spark
from pyspark.sql import Window


def test_at_least_n_distinct(spark_context):
    df = spark_context.parallelize([
        ('foo', 'bar', 'baz'),
        ('foo', 'bar', 'bang'),
        ('foo', 'bar', 'bang'),
        ('foo', 'test', 'test'),
        ('foo', 'test', 'test'),
        ('fizz', 'bang', 'boom'),
    ]).toDF(['a', 'b', 'c'])

    w = Window.partitionBy('a', 'b')
    res = df.withColumn('z', mjolnir.spark.at_least_n_distinct('c', 2).over(w)).collect()
    expect = [
        ('foo', 'bar', 'baz', True),
        ('foo', 'bar', 'bang', True),
        ('foo', 'bar', 'bang', True),
        ('foo', 'test', 'test', False),
        ('foo', 'test', 'test', False),
        ('fizz', 'bang', 'boom', False),
    ]
    assert sorted(map(tuple, res)) == sorted(expect)
