from contextlib import contextmanager
from itertools import islice
import json
import mjolnir.training.xgboost
from mjolnir.utilities.data_pipeline import run_pipeline as run_data_pipeline
from mjolnir.utilities.collect_features import collect_features
from mjolnir.utilities.feature_selection import run_pipeline as run_feature_selection_pipeline
from mjolnir.utilities.make_folds import make_folds
from mjolnir.utilities.training_pipeline import run_pipeline as run_train_pipeline
import os
import pickle
from pyspark.sql import types as T
import pytest
import random
import shutil
import tempfile
import threading
import xgboost


# Expected shape of the query_clicks_daily data
INPUT_SCHEMA = T.StructType([
    T.StructField("wikiid", T.StringType()),
    T.StructField("q_by_ip_day", T.IntegerType()),
    T.StructField("query", T.StringType()),
    T.StructField("session_id", T.StringType()),
    T.StructField("hits", T.ArrayType(T.StructType([
        T.StructField("pageid", T.IntegerType())
    ]))),
    T.StructField("clicks", T.ArrayType(T.StructType([
        T.StructField("pageid", T.IntegerType())
    ]))),
])

# queries sourced from morelike:mediawiki. By using real data (stored in a
# fixture) training generates actual predictions, as opposed to with random
# feature data xgboost mostly predicted 0.5.
# Fixture recorded by using ssh tunnels between mjolnir vagrant instance and
# relforge.
PAGE_IDS = [
  4181610, 323710, 1550921, 28115074, 41672405,
  39834209, 197825, 26296277, 18618509, 2112462,
  21103176, 2007631, 399930, 42933137, 328159,
  17698784, 5043734, 25958124, 179849, 2920561,
  4698944, 29753790, 12028935, 156658, 5314244,
  14072, 33965883, 33957176, 46639550, 1569036,
  2058763, 27751, 68229, 212390, 38385607
]

QUERIES = ["mediawiki", "user", "wiki", "template", "extension"]


def make_fake_row(r, session_id):
    hits = sorted(PAGE_IDS, key=lambda _: r.random())[:5]
    n_clicks = 1 if r.random() > 0.3 else 2
    clicks = sorted(hits, key=lambda _: r.random())[:n_clicks]
    return (
        "enwiki",  # wikiid
        5,  # q_by_ip_day,
        r.choice(QUERIES),
        session_id,
        [{"pageid": pageid} for pageid in hits],
        [{"pageid": pageid} for pageid in clicks])


def make_fake_rows(seed=0):
    r = random.Random(seed)
    session_id = 0
    while True:
        yield make_fake_row(r, session_id)
        if r.random() > 0.4:
            session_id += 1


@contextmanager
def tempdir():
    dir = tempfile.mkdtemp()
    try:
        yield dir
    finally:
        shutil.rmtree(dir)


def wrap_mutex(fn):
    mutex = threading.Lock()

    def inner(*args, **kwargs):
        with mutex:
            return fn(*args, **kwargs)
    return inner


@contextmanager
def xgboost_mutex():
    old_train = mjolnir.training.xgboost.train
    mjolnir.training.xgboost.train = wrap_mutex(old_train)
    try:
        yield
    finally:
        mjolnir.training.xgboost.train = old_train


@pytest.mark.skip(reason="Fails with `Rabit call after finalize` too often in CI")
def test_integration(spark_context, hive_context, make_requests_session):
    """Happy path end-to-end test"""
    def session_factory():
        return make_requests_session('requests/test_integration.sqlite3')

    with tempdir() as dir, xgboost_mutex():
        input_dir = os.path.join(dir, 'input')
        labeled_dir = os.path.join(dir, 'labeled')
        collect_dir = os.path.join(dir, 'features')
        feature_sel_dir = os.path.join(dir, 'pruned')
        folds_dir = os.path.join(dir, 'folded')
        trained_dir = os.path.join(dir, 'trained')

        # Generate some fake sessions and write them out
        hive_context.createDataFrame(
            spark_context.parallelize(islice(make_fake_rows(), 0, 2000)),
            INPUT_SCHEMA
        ).write.parquet(input_dir)

        # Apply data collection to those sessions.
        run_data_pipeline(
            spark_context, hive_context, input_dir, labeled_dir,
            wikis=["enwiki"], samples_per_wiki=5000,
            min_sessions_per_query=1, search_cluster='localhost',
            brokers=None, samples_size_tolerance=0.5,
            session_factory=session_factory)

        # Collect features for the labeled dataset
        # When building the fixture the featureset has to actually exist on
        # whatever elasticsearch is serving up results.
        collect_features(
            spark_context, hive_context, labeled_dir, collect_dir,
            wikis=['enwiki'], search_cluster='localhost',
            brokers=None, ltr_feature_definitions='featureset:enwiki_v1',
            session_factory=session_factory)

        # Run feature selection
        run_feature_selection_pipeline(
            spark_context, hive_context, input_dir=collect_dir, output_dir=feature_sel_dir,
            algo='mrmr', num_features=10, pre_selected=None, wikis=None)

        # Generate folds to feed into training
        make_folds(
            spark_context, hive_context, feature_sel_dir, folds_dir,
            wikis=["enwiki"], zero_features=None, num_folds=2,
            num_workers=1, max_executors=2)

        with open(os.path.join(folds_dir, 'stats.json'), 'r') as f:
            stats = json.load(f)

        # Train a model
        # TODO: training pipeline differs in that it expects the
        # directory to be created by the caller.
        os.mkdir(trained_dir)
        run_train_pipeline(
            spark_context, hive_context, folds_dir, trained_dir,
            wikis=["enwiki"], initial_num_trees=10, final_num_trees=None,
            num_cv_jobs=1, iterations=3)

        with open(os.path.join(trained_dir, 'tune_enwiki.pickle'), 'rb') as f:
            tune = pickle.load(f)

        model_file = os.path.join(trained_dir, 'model_enwiki.xgb')
        pybooster = xgboost.Booster()
        pybooster.load_model(model_file)

        # [5:] trims file: off the beginning
        dmat = xgboost.DMatrix(stats['wikis']['enwiki']['all'][0]['all'][5:] + ".xgb")
        eval_dmat = float(pybooster.eval(dmat).split(':')[1])

        expect = tune['metrics']['train'][-1]
        assert expect == pytest.approx(eval_dmat, abs=0.0001)

        # We have to coalesce(1) because for tests all executors run in same
        # JVM and it isn't thread safe to call into xgboost from multiple
        # executor threads in parallel.
        # model = XGBoostModel.loadModelFromLocalFile(spark_context, model_file)
        # df_data = hive_context.read.parquet(data_dir)
        # eval_df = model.eval(df_data.coalesce(1))

        # Our fake data has a hard time passing this test, because ndcg
        # in xgboost is unstable when multiple observations in the same
        # query have the same predicted score. This should only be a
        # problem when using randomly generated clicks to get labels.
        # assert expect == pytest.approx(eval_df, abs=0.0001)
