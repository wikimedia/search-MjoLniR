"""
Group queries together by intent.

Attempts to group together queries with different textual
input, but that have roughly the same intent. This helps the
DBN layer have more data to learn from.

It works primarily by doing a brute-force grouping using a lucene
stemmer to find queries that are plausibly similar to each other.
Next it collects the top 5 hit pages from elasticsearch for each
query and groups together queries that have very similar result
sets.
"""

import mjolnir.es_hits
import mjolnir.spark
import numpy as np
import pyspark.sql.types
from pyspark.sql import functions as F
from pyspark.sql import Window
import requests


def _batch(a, b, size):
    assert len(a) == len(b)
    idx = 0
    while idx < len(a):
        yield a[idx:idx+size], b[idx:idx+size]
        idx += size


def _binary_sim(matrix):
    """Compute a jaccard similarity matrix.

    Vecorization based on: https://stackoverflow.com/a/40579567

    Parameters
    ----------
    matrix : np.array

    Returns
    -------
    np.array
        matrix of shape (n_rows, n_rows) giving the similarity
        between rows of the input matrix.
    """
    # Generate the indices of the lower triangle of our result matrix.
    # The diagonal is offset by -1 because the identity in a similarity
    # matrix is always 1.
    r, c = np.tril_indices(matrix.shape[0], -1)

    # Particularly large groups can blow out memory usage. Chunk the calculation
    # into steps that require no more than ~100MB of memory at a time.
    # We have 4 2d intermediate arrays in memory at a given time, plus the
    # input and output:
    #  p1 = max_rows * matrix.shape[1]
    #  p2 = max_rows * matrix.shape[1]
    #  intersection = max_rows * matrix.shape[1] * 4
    #  union = max_rows * matrix.shape[1] * 8
    # This adds up to:
    #  memory usage = max_rows * matrix.shape[1] * 14
    mem_limit = 100 * pow(2, 20)
    max_rows = mem_limit / (14 * matrix.shape[1])
    out = np.eye(matrix.shape[0])
    for c_batch, r_batch in _batch(c, r, max_rows):
        # Use those indices to build two matrices that contains all
        # the rows we need to do a similarity comparison on
        p1 = matrix[c_batch]
        p2 = matrix[r_batch]
        # Run the main jaccard calculation
        intersection = np.logical_and(p1, p2).sum(1)
        union = np.logical_or(p1, p2).sum(1).astype(np.float64)
        # Build the result matrix with 1's on the diagonal
        # Insert the result of our similarity calculation at their original indices
        out[c_batch, r_batch] = intersection / union
    # Above only populated half of the matrix, the mirrored diagonal contains
    # only zeros. Fix that up by transposing. Adding the transposed matrix double
    # counts the diagonal so subtract that back out. We could skip this step and
    # leave half the matrix empty, but it takes a fraction of a ms to be correct
    # even on mid-sized inputs (~200 queries).
    return out + out.T - np.diag(np.diag(out))


def _make_query_groups(source, threshold=0.5):
    """Cluster together queries in source based on similarity of result sets

    Parameters
    ----------
    source : list
    threshold : float

    Returns
    -------
    list
    """
    # sets don't have a defined order, so make it a list which has explicit ordering
    # This list will be our set of columns in the matrix
    all_page_ids = list(set([page_id for row in source for page_id in row.hit_page_ids]))
    n_rows = len(source)
    n_columns = len(all_page_ids)
    # No hits? something odd ... but ok. Return a unique
    # group for each query.
    if n_columns == 0:
        return zip([row.query for row in source], range(n_rows))

    # Build up a matrix that has a unique query
    # for each row, and the columns are all known page ids. Cells are set
    # to True if that page id was shown for that query.
    matrix = np.empty((n_rows, n_columns), dtype=np.bool)
    for i, row in enumerate(source):
        hit_page_ids = row.hit_page_ids
        for j, col_page_id in enumerate(all_page_ids):
            matrix[i][j] = col_page_id in hit_page_ids

    # Calculate jaccard similiarity for all combinations of queries. This could
    # get very expensive for large matrices, but since we pre-grouped with the
    # lucene stemmer the size should be reasonable enough.
    sim = _binary_sim(matrix)

    # Perform a very simple clustering of the similarities. There are probably
    # better algorithms for this, although in a test agglomerative clustering
    # returned very similar results.
    # Assigns each row to a unique group id
    groups = range(n_rows)
    # Walk over all the rows
    for i in range(n_rows):
        # The similarity matrix is mirrored over the diagonal, so we only
        # need to visit j up to i
        for j in range(i):
            if sim[i][j] > threshold:
                # Re-assign all items with group[i] to group[j]
                old_group = groups[i]
                new_group = groups[j]
                # As we have only visited up to i, nothing > i can have
                # old_group. range returns 0 to n-1, so we need i+1
                for k in range(0, i+1):
                    if groups[k] == old_group:
                        groups[k] = new_group
    return zip([row.query for row in source], groups)


def transform(df, url_list, indices=None, batch_size=30, top_n=5, min_sessions_per_query=35,
              session_factory=requests.Session):
    """Group together similar results in df

    Attaches a query_id and norm_query_id field to df. query_id uniquely identifies
    a single query string to a single wiki. norm_query_id identifies clusters of similar
    queries based on an initial grouping via the lucene stemmer, and then clustering
    by similarity of result sets.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    url_list : list of str
        List of urls for elasticsearch servers
    indices : dict, optional
        Map from wikiid to the elasticsearch index to query. If a wiki is not provided
        the wikiid will be used as the index name.
    batch_size : int
        Number of queries to issue in a single multi-search
    top_n : int
        Number of hits to collect per query for building query groups.
    min_sessions_per_query : int
        The minimum number of sessions a query group must contain.
    session_factory : object

    Returns
    -------
    pyspark.sql.DataFrame
    """
    mjolnir.spark.assert_columns(df, ['wikiid', 'query', 'session_id'])

    # This UDF must not be created at the top level, or simply including this file
    # will magic up a SparkContext and prevent top level code from setting specific
    # configuration.
    _make_query_groups_udf = F.udf(_make_query_groups, pyspark.sql.types.ArrayType(
        pyspark.sql.types.StructType([
            pyspark.sql.types.StructField("query", pyspark.sql.types.StringType()),
            pyspark.sql.types.StructField("norm_query_group_id", pyspark.sql.types.IntegerType()),
        ])))

    w_norm_q = Window.partitionBy('wikiid', 'norm_query')
    # Build up a dataframe of (wikiid, query, query_id, norm_query_id).
    df_queries = (
        df
        .select('wikiid', 'query', 'session_id',
                F.expr('stemmer(query, substring(wikiid, 1, 2))').alias('norm_query'))
        # Pre-filter low volume norm_query to reduce the number of external queries
        # we need to perform to collect page ids. Note that this intentionally creates
        # a new column, and then checks, rather than in a single step because that triggers
        # an UnsupportedOperationException
        .withColumn('has_min_sessions',
                    mjolnir.spark.at_least_n_distinct('session_id', min_sessions_per_query).over(w_norm_q))
        .where(F.col('has_min_sessions'))
        # Drop all the duplicates from session id's
        .select('wikiid', 'query', 'norm_query')
        .drop_duplicates())

    df_norm_query_id = (
        # Collect the current top hits for each individual query. This is used,
        # as opposed to historical hit page ids because the shift in results over
        # time negatively effects the grouping.
        mjolnir.es_hits.transform(df_queries, url_list, indices, batch_size, top_n, session_factory)
        # Build a row per normalized query
        .groupBy('wikiid', 'norm_query')
        .agg(F.collect_list(F.struct('query', 'hit_page_ids')).alias('source'))
        # This explode basically undoes the collect_list above after attaching
        # group information to each row using a binary vector similarity metric
        .select('wikiid', 'norm_query',
                F.explode(_make_query_groups_udf('source')).alias('norm_query_group'))
        # Move the fields of norm_query_group to top level columns
        .select('wikiid', 'norm_query', 'norm_query_group.*')
        # Re-group by the new groups so we can give them unique ids
        .groupBy('wikiid', 'norm_query', 'norm_query_group_id')
        .agg(F.collect_list('query').alias('queries'))
        .select('wikiid', 'queries', F.monotonically_increasing_id().alias('norm_query_id'))
        # Expand back out into a df with columns (query_id, norm_query_id)
        .select('wikiid', F.explode('queries').alias('query'), 'norm_query_id'))

    w_norm_q_id = Window.partitionBy('norm_query_id')
    return (
        df
        .join(df_norm_query_id, how='inner', on=['wikiid', 'query'])
        # Re-apply minimum number of sessions per group, now that we have final groupings.
        .withColumn('has_min_sessions',
                    mjolnir.spark.at_least_n_distinct('session_id', min_sessions_per_query).over(w_norm_q_id))
        .where(F.col('has_min_sessions'))
        .drop('has_min_sessions'))
