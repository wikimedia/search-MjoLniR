from __future__ import absolute_import
import findspark
findspark.init()  # must happen before importing pyspark

import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
import pytest  # noqa: E402
import requests  # noqa: E402
import sqlite3  # noqa: E402


def quiet_log4j():
    logger = logging.getLogger('py4j')
    logger.setLevel(logging.WARN)


@pytest.fixture
def fixtures_dir():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cur_dir, 'fixtures')


@pytest.fixture
def folds_a(fixtures_dir):
    fixtures_dir = os.path.join(fixtures_dir, 'datasets')
    return [{
        "train": os.path.join(fixtures_dir, "train.xgb"),
        "test": os.path.join(fixtures_dir, "test.xgb")
    }]


@pytest.fixture
def folds_b(fixtures_dir):
    fixtures_dir = os.path.join(fixtures_dir, 'datasets')

    def f(path):
        return os.path.join(fixtures_dir, path + ".xgb")

    return [
        {"train": f("train.f0.p0")},
        {"train": f("train.f1.p0")}
    ]


@pytest.fixture(scope="session")
def spark():
    """Fixture for creating a spark context.

    Args:
        request: pytest.FixtureRequest object

    Returns:
        SparkContext for tests
    """

    quiet_log4j()
    # Pull appropriate jvm dependencies from archiva. Would be nice
    # if we could provide this in SparkConf, but in 2.1.x there isn't
    # a way.
    os.environ['PYSPARK_SUBMIT_ARGS'] = '--repositories %s pyspark-shell' % (
            ','.join(['https://archiva.wikimedia.org/repository/%s' % (repo)
                      for repo in ['releases', 'snapshots', 'mirrored']]))

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    builder = (
        SparkSession.builder
        .master("local[2]")
        .appName("pytest-pyspark-local-testing")
        # Local jvm package. Compiled by the tox `jvm` testenv
        # TODO: Source version number from somewhere.
        .config('spark.jars', os.path.join(base_dir, 'jvm/target/mjolnir-0.5-SNAPSHOT.jar'))
        # Maven coordinates of jvm dependencies
        .config('spark.jars.packages', 'sramirez:spark-infotheoretic-feature-selection:1.4.4')
        # By default spark will shuffle to 200 partitions, which is
        # way too many for our small test cases. This cuts execution
        # time of the tests in half.
        .config('spark.sql.shuffle.partitions', 4)
    )
    if 'XDG_CACHE_HOME' in os.environ:
        builder.config('spark.jars.ivy', os.path.join(os.environ['XDG_CACHE_HOME'], 'ivy2'))

    with builder.getOrCreate() as spark:
        yield spark


@pytest.fixture(scope="session")
def spark_context(spark):
    return spark.sparkContext


@pytest.fixture()
def make_requests_session(fixtures_dir):
    def f(path):
        if path[0] != '/':
            path = os.path.join(fixtures_dir, path)
        return MockSession(path)
    return f


class MockSession(object):
    def __init__(self, fixture_file):
        self._session = None
        # Use sqlite for storage so we don't have to figure out how
        # multiple pyspark executors write to the same file
        self.sqlite = sqlite3.connect(fixture_file)
        self.sqlite.execute(
            "CREATE TABLE IF NOT EXISTS requests " +
            "(digest text PRIMARY KEY, status_code int, content text)")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get(self, url, data=None):
        return self.request('GET', url, data=data)

    def request(self, method, url, data=None, headers=None):
        # headers parameter intentionall ignored.
        # TODO: This hash is far from a uniqueness guarantee, it's
        # not really even best effort in the face of hash collisions
        md5 = hashlib.md5()
        if method.upper() != 'GET':
            # Only add non-GET method to md5 to keep hashes from when
            # that was the only option.
            md5.update(method.encode('utf-8'))
        md5.update(url.encode('utf-8'))
        if data is not None:
            md5.update(data.encode('utf-8'))
        digest = md5.hexdigest()

        for row in self.sqlite.execute("SELECT status_code, content from requests WHERE digest=?", [digest]):
            return MockResponse(row[0], row[1])

        try:
            r = requests.request(method, url, data=data)
        except requests.exceptions.ConnectionError:
            logging.exception("%s %s\n%s", method, url, data)
            raise

        try:
            self.sqlite.execute("INSERT INTO requests VALUES (?,?,?)", [digest, r.status_code, r.text])
            self.sqlite.commit()
        except sqlite3.IntegrityError:
            # inserted elsewhere? no big deal
            pass

        return MockResponse(r.status_code, r.text)


class MockResponse(object):
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text

    def json(self):
        return json.loads(self.text)
