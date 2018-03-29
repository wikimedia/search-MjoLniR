"""
Calculates NDCG@k values for click data
"""

from __future__ import absolute_import
import math
from pyspark.sql import functions as F
from pyspark.sql import Window
import pyspark.sql.types


def _ndcg_at(k, label_col):
    def ndcg_at_k(predicted, actual):
        # TODO: Taking in rn and then re-sorting might not be necessary, but i can't
        # find any real guarantee that they would come in order after a groupBy + collect_list,
        # since they were only ordered within the window function.
        predicted = [row[label_col] for row in sorted(predicted, key=lambda r: r.rn)]
        actual = [row[label_col] for row in sorted(actual, key=lambda r: r.rn)]
        dcg = 0.
        for i, label in enumerate(predicted):
            # This form is used to match EvalNDCG in xgboost
            dcg += ((1 << label) - 1) / math.log(i + 2.0, 2)
        idcg = 0.
        for i, label in enumerate(actual):
            idcg += ((1 << label) - 1) / math.log(i + 2.0, 2)
        if idcg == 0:
            return 0
        else:
            return dcg / idcg
    return F.udf(ndcg_at_k, pyspark.sql.types.DoubleType())


def ndcg(df, k, label_col='label', position_col='hit_position', wiki_col='wikiid',
         query_cols=['wikiid', 'query', 'session_id']):
    """
    Calculate ndcg@k for the provided dataframe

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input dataframe to calculate against
    k : int
        Cutoff for ndcg calculation
    label_col : str
        Column name containing integer label, higher is better, of the hit
    position_col : str
        Column name containing order displayed to user, lowest first, of the hit
    query_cols : list of str
        Column names to group by, which indicate a unique query displayed to a user

    Returns
    -------
    float
        The ndcg@k value, always between 0 and 1
    """
    if wiki_col not in query_cols:
        query_cols = query_cols + [wiki_col]

    # ideal results per labels
    w = Window.partitionBy(*query_cols).orderBy(F.col(label_col).desc())
    topAtK = (
        df
        .select(label_col, *query_cols)
        .withColumn('rn', F.row_number().over(w))
        .where(F.col('rn') <= k)
        .groupBy(*query_cols)
        .agg(F.collect_list(F.struct(label_col, 'rn')).alias('topAtK')))
    # top k results shown to user
    w = Window.partitionBy(*query_cols).orderBy(F.col(position_col).asc())
    predictedTopAtK = (
        df
        .select(label_col, position_col, *query_cols)
        .withColumn('rn', F.row_number().over(w))
        .where(F.col('rn') <= k)
        .groupBy(*query_cols)
        .agg(F.collect_list(F.struct(label_col, 'rn')).alias('predictedTopAtK')))
    return {row[wiki_col]: row.ndcgAtK for row in topAtK
            .join(predictedTopAtK, query_cols, how='inner')
            .select(wiki_col, _ndcg_at(k, label_col)('predictedTopAtK', 'topAtK').alias('ndcgAtK'))
            .groupBy(wiki_col)
            .agg(F.mean('ndcgAtK').alias('ndcgAtK'))
            .collect()}
