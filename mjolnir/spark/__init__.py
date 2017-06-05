"""
Helper functions for dealing with pyspark
"""
import json
from pyspark import SparkContext
from pyspark.sql import Column, functions as F
from pyspark.sql.column import _to_java_column, _to_seq


def assert_columns(df, columns):
    """ Raise an exception if the dataframe
    does not contain the desired columns
    Parameters
    ----------
    df : pyspark.sql.DataFrame
    columns : list of strings
        Set of columns that must be present in df
    """
    have = set(df.columns)
    need = set(columns)
    if not need.issubset(have):
        raise ValueError("Missing columns in DataFrame: %s" % (", ".join(need.difference(have))))


def add_meta(sc, col, metadata):
    """Add metadata to a column

    Adds metadata to a column for describing extra properties. This metadata survives
    serialization from dataframe to parquet and back to dataframe. Any manipulation
    of the column, such as aliasing, will lose the metadata.

    Parameters
    ----------
    sc : pyspark.SparkContext
    col : pyspark.sql.Column
    metadata : dict

    Returns
    -------
    pyspark.sql.Column
    """
    meta = sc._jvm.org.apache.spark.sql.types \
        .Metadata.fromJson(json.dumps(metadata))
    return Column(getattr(col._jc, 'as')('', meta))


def at_least_n_distinct(col, limit):
    """Count distinct that works with windows

    The standard distinct count in spark sql can't be applied in
    a window. This implementation allows that to work
    """
    sc = SparkContext._active_spark_context
    j_cols = _to_seq(sc, [_to_java_column(col), _to_java_column(F.lit(limit))])
    jc = sc._jvm.org.wikimedia.search.mjolnir.AtLeastNDistinct().apply(j_cols)
    return Column(jc)
