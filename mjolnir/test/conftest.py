import findspark
findspark.init()

import pytest
import logging
from pyspark import SparkContext, SparkConf
from pyspark.sql import HiveContext


def quiet_log4j():
    logger = logging.getLogger('py4j')
    logger.setLevel(logging.WARN)


@pytest.fixture(scope="session")
def spark_context(request):
    """Fixture for creating a spark context.

    Args:
        request: pytest.FixtureRequest object

    Returns:
        SparkContext for tests
    """
    quiet_log4j()
    conf = (
        SparkConf()
        .setMaster("local[2]")
        .setAppName("pytest-pyspark-local-testing"))
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
