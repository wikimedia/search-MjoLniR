"""
Helper functions for dealing with pyspark
"""
import json
from pyspark.sql import Column


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
