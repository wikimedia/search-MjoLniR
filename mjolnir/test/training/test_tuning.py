import hyperopt
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
                (F.col('id')/100).cast('int').alias('norm_query')))

    with_folds = mjolnir.training.tuning.split(df, (0.8, 0.2), num_partitions=4).collect()

    fold_0 = [row for row in with_folds if row.fold == 0]
    fold_1 = [row for row in with_folds if row.fold == 1]

    # Check the folds are pretty close to requested
    total_len = float(len(with_folds))
    assert 0.8 == pytest.approx(len(fold_0) / total_len, abs=0.015)
    assert 0.2 == pytest.approx(len(fold_1) / total_len, abs=0.015)

    # Check each norm query is only found on one side of the split
    queries_in_0 = set([row.norm_query for row in fold_0])
    queries_in_1 = set([row.norm_query for row in fold_1])
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
    ).toDF(['wikiid', 'norm_query', 'query', 'label', 'features'])


def test_cross_validate_plain_df(df_train):
    scores = mjolnir.training.tuning.cross_validate(
        df_train,
        mjolnir.training.xgboost.train,
        {'objective': 'rank:ndcg', 'eval_metric': 'ndcg@3', 'num_rounds': 1},
        # xgboost needs all jobs to have a worker assigned before it will
        # finish a round of training, so we have to be careful not to use
        # too many workers
        num_folds=2, num_fold_partitions=1, num_cv_jobs=1, num_workers=1)
    # one score for each fold
    assert len(scores) == 2


def test_hyperopt(df_train):
    "Not an amazing test...basically sees if the happy path doesnt blow up"
    space = {
        'num_rounds': 50,
        'max_depth': hyperopt.hp.quniform('max_depth', 1, 20, 1)
    }

    # mostly hyperopt just calls cross_validate, of which the integration with
    # xgboost is separately tested. Instead of going all the way into xgboost
    # mock it out w/MockModel.
    best_params, trails = mjolnir.training.tuning.hyperopt(
        df_train, MockModel, space, max_evals=5, num_folds=2,
        num_fold_partitions=1, num_cv_jobs=1, num_workers=1)
    assert isinstance(best_params, dict)
    # num_rounds should have been unchanged
    assert 'num_rounds' in best_params
    assert best_params['num_rounds'] == 50
    # should have max_evals evaluations
    assert len(trails.trials) == 5


class MockModel(object):
    def __init__(self, df, params, num_workers):
        # Params that were passed to hyperopt
        assert isinstance(params, dict)
        assert 'max_depth' in params
        assert params['num_rounds'] == 50
        assert num_workers == 1

    def eval(self, df_test, j_groups=None, feature_col='features', label_col='label'):
        return 1.0
