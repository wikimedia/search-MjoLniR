from __future__ import absolute_import
import mjolnir.training.xgboost
from pyspark.ml.linalg import Vectors
import pytest


def _assert_seq_of_seq(expected, j_seqs):
    "Assert Seq[Seq[_]] matches expected"
    assert len(expected) == j_seqs.length()
    for i, expect in enumerate(expected):
        j_seq = j_seqs.apply(i)
        group = [j_seq.apply(j) for j in range(j_seq.length())]
        assert expect == group


@pytest.fixture()
def train_path(fixtures_dir):
    """
    Fake data generated with:

    data = [(random.randint(0,3), [random.random() for i in range(5)]) for i in range(100)]
    with open('train.xgb', 'w') as f:
      for label, features in data:
        f.write("%d %s\n" % (label, " ".join(["%d:%f" % (i+1, feat) for i, feat in enumerate(features)])))
    with open('train.xgb.query', 'w') as f:
       i = 0
       while i < len(data):
           next = min(len(data) - i, random.randint(10,20))
           f.write("%d\n" % (next))
           i += next
    """
    return fixtures_dir + '/datasets/train.xgb'


@pytest.fixture()
def test_path(fixtures_dir):
    return fixtures_dir + '/datasets/test.xgb'


@pytest.fixture()
def df_train(spark_context):
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


def test_train_minimum_params(df_train, folds_a):
    params = {'num_rounds': 1}
    model = mjolnir.training.xgboost.train(folds_a[0], params, 'train')

    assert isinstance(model, mjolnir.training.xgboost.XGBoostModel)
    # make sure train didn't clobber the incoming params
    assert len(params) == 1
    assert params['num_rounds'] == 1
