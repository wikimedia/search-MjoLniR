from __future__ import absolute_import
import findspark
findspark.init()  # must happen before importing pyspark

import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
from pyspark import SparkContext, SparkConf  # noqa: E402
from pyspark.sql import HiveContext  # noqa: E402
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


@pytest.fixture(scope="session")
def spark_context(request):
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
    conf = (
        SparkConf()
        .setMaster("local[2]")
        .setAppName("pytest-pyspark-local-testing")
        # Maven coordinates of jvm dependencies
        .set('spark.jars.packages', ','.join([
            'ml.dmlc:xgboost4j-spark:0.7-wmf-1',
            'org.wikimedia.search:mjolnir:0.2',
            'org.apache.spark:spark-streaming-kafka-0-8_2.11:2.1.0']))
        # By default spark will shuffle to 200 partitions, which is
        # way too many for our small test cases. This cuts execution
        # time of the tests in half.
        .set('spark.sql.shuffle.partitions', 4))
    sc = SparkContext(conf=conf)
    yield sc
    sc.stop()


@pytest.fixture(scope="session")
def hive_context(spark_context):
    """Fixture for creating a Hive context.

    Args:
        spark_context: spark_context fixture

    Returns:
        HiveContext for tests
    """
    return HiveContext(spark_context)


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
        md5 = hashlib.md5()
        md5.update(url)
        if data is not None:
            md5.update(data)
        digest = md5.hexdigest()

        for row in self.sqlite.execute("SELECT status_code, content from requests WHERE digest=?", [digest]):
            return MockResponse(row[0], row[1])

        r = requests.get(url, data=data)

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
