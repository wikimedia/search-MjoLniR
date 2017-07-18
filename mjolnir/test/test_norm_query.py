from collections import namedtuple
import mjolnir.norm_query
import numpy as np
import os
import pytest


@pytest.fixture
def df_love(spark_context, hive_context, fixtures_dir):
    path = os.path.join(fixtures_dir, 'love.json')
    return hive_context.read.json(path)


def test_norm_query(df_love, hive_context, make_requests_session):
    """Very basic happy path test of query normalization"""
    def session_factory():
        return make_requests_session('requests/test_norm_query.sqlite3')

    # Make a fake stemmer() udf. We know everything in df_love
    # stems to 'love' because that's how it was generated
    hive_context.registerFunction("stemmer", lambda x, y: "love")

    df_res = mjolnir.norm_query.transform(
        df_love,
        url_list=['http://localhost:9200'],
        min_sessions_per_query=10,
        session_factory=session_factory)

    # 1402 rows of input data made it through the min_session limits
    assert df_res.count() == 1402
    # 6 individual query groups were found
    assert df_res.select('norm_query_id').drop_duplicates().count() == 6
    # Those 6 query groups are made up of 124 different queries
    assert df_res.select('wikiid', 'query', 'norm_query_id').drop_duplicates().count() == 124


@pytest.mark.parametrize("hits, expected", [
    # all overlap, same group
    ([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], [0, 0]),
    # 4 of 5 overlap, same group
    ([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], [0, 0]),
    # 3 of 5 overlap, not grouped
    ([[1, 2, 3, 4, 5], [3, 4, 5, 6, 7]], [0, 1]),
    # 4 of 5 overlap with second item, all same group
    ([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]], [0, 0, 0]),
    # first item doesn't overlap enough to group with the other two
    ([[2, 3, 6, 7, 8], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]], [0, 1, 1]),
    # One item no hits
    ([[1, 2], []], [0, 1]),
    # All items no hits
    ([[], []], [0, 1]),
])
def test_make_query_groups(hits, expected):
    row = namedtuple('row', ('query', 'hit_page_ids'))
    source = [row(str(i), hit_page_ids) for i, hit_page_ids in enumerate(hits)]
    groups = mjolnir.norm_query._make_query_groups(source)
    assert [(str(i), g) for i, g in enumerate(expected)] == groups


def test_vectorized_jaccard_sim():
    # The vectorized version of jaccard similarity is 20x faster, but it is
    # harder to understand. Compute it the simple way and compare to the
    # vectorized version
    def jaccard_sim(X, Y):
        assert len(X) == len(Y)
        a = np.sum((X == 1) & (Y == 1))
        d = np.sum((X == 0) & (Y == 0))
        return a / float(len(X) - d)

    def binary_sim(mat):
        n_rows = mat.shape[0]
        out = np.empty((n_rows, n_rows), dtype=np.float64)
        for i in range(n_rows):
            out[i][i] = 1.
            for j in range(0, i):
                out[i][j] = jaccard_sim(mat[i], mat[j])
                out[j][i] = out[i][j]
        return out

    # Simulate 200 queries with 100 shared page ids
    matrix = np.random.rand(200, 100) > 0.7
    simple = binary_sim(matrix)
    vectorized = mjolnir.norm_query._binary_sim(matrix)
    assert np.array_equal(simple, vectorized)
