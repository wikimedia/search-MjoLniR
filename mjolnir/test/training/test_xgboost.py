import mjolnir.training.xgboost
from pyspark.ml.linalg import Vectors
import pytest


@pytest.fixture()
def df_prep_training(spark_context, hive_context):
    rdd1 = spark_context.parallelize([
        ('foowiki', 'foo', 4, Vectors.dense([1., 2.])),
        ('foowiki', 'foo', 2, Vectors.dense([2., 1.])),
        ('barwiki', 'bar', 4, Vectors.dense([0., 1.])),
    ], 1)
    rdd2 = spark_context.parallelize([
        ('foowiki', 'bar', 1, Vectors.dense([2., 1.])),
        ('barwiki', 'baz', 2, Vectors.dense([2., 0.])),
    ], 1)
    # union together two single partition rdd's to guarnatee
    # shape and content of output partitions
    return (
        spark_context
        .union([rdd1, rdd2])
        .toDF(['wikiid', 'query', 'label', 'features']))


def _assert_seq_of_seq(expected, j_seqs):
    "Assert Seq[Seq[_]] matches expected"
    assert len(expected) == j_seqs.length()
    for i, expect in enumerate(expected):
        j_seq = j_seqs.apply(i)
        group = [j_seq.apply(j) for j in range(j_seq.length())]
        assert expect == group


def test_prep_training_no_params(df_prep_training):
    #  Double check, should always be true
    assert 2 == df_prep_training.rdd.getNumPartitions()
    df_grouped, j_groups = mjolnir.training.xgboost.prep_training(df_prep_training)

    # Makes some very bold assumptions about what spark did behind the scene
    # when repartitioning...at least spark is very deterministic.
    # with no params number of partitions should be unchanged
    # TODO: This doesn't seem right, shouldn't it be [2,1], [1,1]?
    expected = [[1, 2], [1, 1]]
    assert len(expected) == df_grouped.rdd.getNumPartitions()
    _assert_seq_of_seq(expected, j_groups)


def test_prep_training_w_num_workers(df_prep_training):
    num_workers = 1
    df_grouped, j_groups = mjolnir.training.xgboost.prep_training(
        df_prep_training, num_workers)
    expected = [[1, 1, 1, 2]]
    assert num_workers == df_grouped.rdd.getNumPartitions()
    assert len(expected) == num_workers
    _assert_seq_of_seq(expected, j_groups)


@pytest.fixture()
def df_train(spark_context, hive_context):
    rdd1 = spark_context.parallelize([
        ('foowiki', 'foo', 2, Vectors.dense([2.2])),
    ])
    rdd2 = spark_context.parallelize([
        # Use two queries with same features and different
        # values to have ndcg != 1.0
        ('foowiki', 'foo', 4, Vectors.dense([1.1])),
        ('foowiki', 'foo', 1, Vectors.dense([1.1])),
    ])
    return (
        spark_context
        .union([rdd1, rdd2])
        .toDF(['wikiid', 'query', 'label', 'features']))


def test_train_minimum_params(df_train):
    params = {'num_rounds': 1}
    # TODO: Anything > 1 worker can easily get stuck, as a system
    # with 2 cpus will only spawn a single worker.
    model = mjolnir.training.xgboost.train(df_train, params, num_workers=1)

    # What else can we practically assert?
    df_transformed = model.transform(df_train)
    assert 'prediction' in df_transformed.columns
    assert 0.74 == pytest.approx(model.eval(df_train), abs=0.01)

    # make sure train didn't clobber the incoming params
    assert params['num_rounds'] == 1


def _always_raise(*args, **kwargs):
    raise ValueError('should not be called')


def test_train_pre_prepped(df_train):
    num_workers = 1
    params = {'num_rounds': 1}

    df_grouped, j_groups = mjolnir.training.xgboost.prep_training(
        df_train, num_workers)
    params['groupData'] = j_groups

    # TODO: This is probably not how we should make sure it isn't called..
    orig_prep_training = mjolnir.training.xgboost.prep_training
    try:
        mjolnir.training.xgboost.prep_training = _always_raise
        model = mjolnir.training.xgboost.train(df_grouped, params)
        assert 0.74 == pytest.approx(model.eval(df_grouped, j_groups), abs=0.01)
    finally:
        mjolnir.training.xgboost.prep_training = orig_prep_training
