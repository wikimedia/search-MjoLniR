from __future__ import absolute_import
import mjolnir.training.tuning
import mjolnir.training.xgboost
from pyspark.sql import functions as F
from pyspark.ml.linalg import Vectors
import pytest


def test_split(spark_context, hive_context):
    df = (
        hive_context
        .range(1, 100 * 100)
        # convert into 100 "queries" with 100 values each. We need a
        # sufficiently large number of queries, or the split wont have
        # enough data for partitions to even out.
        .select(F.lit('foowiki').alias('wikiid'),
                (F.col('id')/100).cast('int').alias('norm_query_id')))

    with_folds = mjolnir.training.tuning.split(df, (0.8, 0.2), num_partitions=4).collect()

    fold_0 = [row for row in with_folds if row.fold == 0]
    fold_1 = [row for row in with_folds if row.fold == 1]

    # Check the folds are pretty close to requested
    total_len = float(len(with_folds))
    assert 0.8 == pytest.approx(len(fold_0) / total_len, abs=0.015)
    assert 0.2 == pytest.approx(len(fold_1) / total_len, abs=0.015)

    # Check each norm query is only found on one side of the split
    queries_in_0 = set([row.norm_query_id for row in fold_0])
    queries_in_1 = set([row.norm_query_id for row in fold_1])
    assert len(queries_in_0.intersection(queries_in_1)) == 0


def _make_q(query, n=4):
    "Generates single feature queries"
    return [('foowiki', query, query, float(f), Vectors.dense([float(f)])) for f in range(n)]


@pytest.fixture
def df_train(spark_context, hive_context):
    # TODO: Use some fixture dataset representing real-ish data? But
    # it needs to be pretty small
    return spark_context.parallelize(
        _make_q('abc') + _make_q('def') + _make_q('ghi') + _make_q('jkl')
        + _make_q('mno') + _make_q('pqr') + _make_q('stu')
    ).toDF(['wikiid', 'norm_query_id', 'query', 'label', 'features'])


def test_cross_validate_plain_df(df_train):
    scores = mjolnir.training.tuning.cross_validate(
        df_train,
        mjolnir.training.xgboost.train,
        {'objective': 'rank:ndcg', 'eval_metric': 'ndcg@3', 'num_rounds': 1},
        # xgboost needs all jobs to have a worker assigned before it will
        # finish a round of training, so we have to be careful not to use
        # too many workers
        num_folds=2, num_workers=1, pool=None)
    # one score for each fold
    assert len(scores) == 2
