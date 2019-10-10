from __future__ import absolute_import
import hyperopt
import mjolnir.training.tuning
import mjolnir.training.xgboost
from pyspark.sql import functions as F
import pytest


def test_split(spark):
    df = (
        spark
        .range(1, 100 * 100)
        # convert into 100 "queries" with 100 values each. We need a
        # sufficiently large number of queries, or the split wont have
        # enough data for partitions to even out.
        .select(F.lit('foowiki').alias('wikiid'),
                (F.col('id')/100).cast('int').alias('norm_query_id')))

    with_folds = mjolnir.training.tuning.split(df, (0.8, 0.2)).collect()

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


def run_model_selection(tune_stages, f=None, num_cv_jobs=1, **kwargs):
    stats = {'called': 0}
    initial_space = {'foo': 10, 'bar': 20, 'baz': 0}
    folds = [3, 6]
    if not f:
        def f(fold, params, **kwargs):
            stats['called'] += 1
            factor = 1.0 / (6 * params['foo'])
            return {
                'test': fold * factor * 0.9,
                'train': fold * factor,
            }

    tuner = mjolnir.training.tuning.ModelSelection(initial_space, tune_stages)
    train_func = mjolnir.training.tuning.make_cv_objective(f, folds, num_cv_jobs, **kwargs)
    trials_pool = tuner.build_pool(folds, num_cv_jobs)
    result = tuner(train_func, trials_pool)
    return result, stats['called']


def test_ModelSelection():
    num_iterations = 3
    result, called = run_model_selection([
        ('a', {
            'iterations': num_iterations,
            'space': {
                'foo': hyperopt.hp.uniform('foo', 1, 9),
            },
        }),
        ('b', {
            'iterations': num_iterations,
            'space': {
                'bar': hyperopt.hp.uniform('bar', 1, 5),
            },
        })
    ])
    # stages * iterations * folds
    assert called == 2 * num_iterations * 2
    # We should still have three parameters
    assert len(result['params']) == 3
    # foo should have a new value between 1 and 9
    assert 1 <= result['params']['foo'] <= 9
    # bar should have a new value between 1 and 5
    assert 1 <= result['params']['bar'] <= 5
    # baz should be untouched
    assert result['params']['baz'] == 0


def test_ModelSelection_kwargs_pass_thru():
    expected_kwargs = {'hi': 5, 'there': 'test'}

    def f(fold, params, **kwargs):
        assert kwargs == expected_kwargs
        return {'test': [fold[0]], 'train': [fold[0]]}

    obj = mjolnir.training.tuning.make_cv_objective(f, [[1], [2]], 1, **expected_kwargs)

    res = obj(None)
    assert res == [
        {'test': [1], 'train': [1]},
        {'test': [2], 'train': [2]}
    ]


@pytest.mark.parametrize(
    "num_folds, num_cv_jobs, expect_pool", [
        (1,           1,       False),
        (1,           2,       True),

        (3,           1,       False),
        (3,           5,       True),
        (3,           6,       True),

        (5,           5,       False),
        (5,           9,       True),
        (5,          11,       True),
    ])
def test_ModelSelection_build_pool(num_folds, num_cv_jobs, expect_pool):
    tuner = mjolnir.training.tuning.ModelSelection(None, None)
    folds = [{} for i in range(num_folds)]
    pool = tuner.build_pool(folds, num_cv_jobs)
    assert (pool is not None) == expect_pool


def test_ModelSelection_transformer():
    stats = {'called': 0}

    def transformer(result, params):
        assert 'foo' in result
        assert result['foo'] == 'bar'
        assert params == 'some params'
        stats['called'] += 1
        return 'baz'

    def f(fold, params):
        assert params == 'some params'
        return {'foo': 'bar'}

    folds = [[1, 2, 3], [4, 5, 6]]
    obj = mjolnir.training.tuning.make_cv_objective(f, folds, 1, transformer)
    assert obj('some params') == ['baz', 'baz']
    assert stats['called'] == 2
