from __future__ import absolute_import
import hyperopt
import mjolnir.training.hyperopt
from pyspark.ml.linalg import Vectors
import pytest


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


def test_minimize(folds_b):
    "Not an amazing test...basically sees if the happy path doesnt blow up"
    space = {
        'num_rounds': 50,
        'max_depth': hyperopt.hp.quniform('max_depth', 1, 20, 1)
    }

    # mostly hyperopt just calls cross_validate, of which the integration with
    # xgboost is separately tested. Instead of going all the way into xgboost
    # mock it out w/MockModel.
    best_params, trails = mjolnir.training.hyperopt.minimize(
        folds_b, MockModel, space, max_evals=5)
    assert isinstance(best_params, dict)
    # num_rounds should have been unchanged
    assert 'num_rounds' in best_params
    assert best_params['num_rounds'] == 50
    # should have max_evals evaluations
    assert len(trails.trials) == 5


class MockSummary(object):
    def train(self):
        return [1.]

    def test(self):
        return [1.]


class MockModel(object):
    def __init__(self, df, params, train_matrix=None):
        # Params that were passed to hyperopt
        assert isinstance(params, dict)
        assert 'max_depth' in params
        assert params['num_rounds'] == 50

    def eval(self, df_test, j_groups=None, feature_col='features', label_col='label'):
        return 1.0

    def summary(self):
        return MockSummary()
