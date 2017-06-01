import mjolnir.metrics
import pytest


def test_ndcg_doesnt_completely_fail(spark_context, hive_context):
    "Mediocre test that just looks for a happy path"
    df = spark_context.parallelize([
        [4, 0, 'foo'],
        [3, 1, 'foo'],
        [0, 2, 'foo'],
        [3, 3, 'foo'],
    ]).toDF(['label', 'hit_position', 'query'])

    # Top 2 are in perfect order. Also this indirectly tests that
    # k is really top 2, and not somehow top 3 or some such
    ndcg_at_2 = mjolnir.metrics.ndcg(df, 2, query_cols=['query'])
    assert 1.0 == ndcg_at_2

    # Top 4 are slightly out. This value was checked by also
    # calculating by hand.
    ndcg_at_4 = mjolnir.metrics.ndcg(df, 4, query_cols=['query'])
    assert 0.9788 == pytest.approx(ndcg_at_4, 0.0001)


def test_query_can_be_multiple_columns(spark_context, hive_context):
    df_a = spark_context.parallelize([
        [4, 0, 'foo', 'bar'],
        [3, 1, 'foo', 'bar'],
        [1, 2, 'foo', 'bar'],
        [2, 3, 'foo', 'bar'],
    ]).toDF(['label', 'hit_position', 'query', 'wiki'])

    df_b = spark_context.parallelize([
        [1, 0, 'foo', 'wot'],
        [3, 1, 'foo', 'wot'],
        [1, 2, 'foo', 'wot'],
        [4, 3, 'foo', 'wot'],
    ]).toDF(['label', 'hit_position', 'query', 'wiki'])

    df_merged = (
        spark_context
        .union([df_a.rdd, df_b.rdd])
        .toDF(['label', 'hit_position', 'query', 'wiki']))

    ndcg_a = mjolnir.metrics.ndcg(df_a, 4, query_cols=['query', 'wiki'])
    ndcg_b = mjolnir.metrics.ndcg(df_b, 4, query_cols=['query', 'wiki'])

    # If we appropriately seperate foo/bar from foo/wot, instead of treating
    # all foo's the same then the result will be the average of the two
    # queries.
    ndcg_merged = mjolnir.metrics.ndcg(df_merged, 4, query_cols=['query', 'wiki'])
    assert ndcg_merged == (ndcg_a + ndcg_b) / 2
