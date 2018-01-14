"""
Generate relevance probabilities from user search sessions.
"""

from __future__ import absolute_import
import pyspark.sql
from pyspark.sql import functions as F, types as T


def train(df, dbn_config):
    """Generate relevance labels for the provided dataframe.

    Process the provided data frame to generate relevance scores for
    all provided pairs of (wikiid, norm_query_id, hit_page_id). The input
    DataFrame must have a row per hit_page_id that was seen by a session.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        User click logs with columns wikiid, norm_query_id, session_id,
        hit_page_id, hit_position, clicked.
    dbn_config : dict
        Configuration needed by the DBN. See scala implementation docs
        for more information.

    Returns
    -------
    spark.sql.DataFrame
        DataFrame with columns wikiid, norm_query_id, hit_page_id, relevance.
    """

    df = (
        df
        .withColumn('hit_page_id', F.col('hit_page_id').cast(T.IntegerType()))
        .withColumn('hit_position', F.col('hit_position').cast(T.IntegerType())))
    jvm = df._sc._jvm
    # jvm side expects Map[String, String]
    j_config = jvm.PythonUtils.toScalaMap({str(k): str(v) for k, v in dbn_config.items()})
    assert j_config.size() == len(dbn_config)
    j_df = jvm.org.wikimedia.search.mjolnir.DBN.train(df._jdf, j_config)
    return pyspark.sql.DataFrame(j_df, df.sql_ctx)
