"""
Helper functions for dealing with pyspark
"""


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
