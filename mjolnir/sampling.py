"""
Takes in the full click logs and samples down to a set appropriate
for running through the MLR pipeline.

Is perhaps overly complex because the utilities in spark aren't
particularly designed to work with multiple datasets (wikis)
in a single dataframe, but it works well enough for our data sizes.
"""

import bisect
import mjolnir.spark
import pyspark
from pyspark.sql import functions as F
from pyspark.sql.column import Column, _to_java_column


def _array_contains(array, value):
    """Generic version of pyspark.sql.functions.array_contains

    array_contains provided by pyspark only allow checking if a value is inside
    a column, but the value has to be a literal and not a column from the row.
    This generalizes the function to allow the value to be a column, checking
    if a column is within a provided literal array.

    >>> df = sc.parallelize([['foo'], ['bar']]).toDF(['id'])
    >>> df.select(_array_contains(F.array(map(F.lit, ['this', 'is', 'foo'])), F.col('id'))).collect()
    [Row(array_contains(array(this,is,foo),id)=True), Row(array_contains(array(this,is,foo),id)=False)]

    Parameters
    ----------
    array : pyspark.sql.Column
    value : pyspark.sql.Column

    Returns
    -------
    pyspark.sql.Column
        Column representing the array_contains expression
    """
    j_array_expr = _to_java_column(array).expr()
    j_value_expr = _to_java_column(value).expr()

    sql = pyspark.SparkContext._active_spark_context._jvm.org.apache.spark.sql
    j_expr = sql.catalyst.expressions.ArrayContains(j_array_expr, j_value_expr)
    jc = sql.Column(j_expr)
    return Column(jc)


def _calc_splits(df, num_buckets=100):
    """Calculate the right edge of num_session buckets

    Utilizes approxQuantile to bucketize the source. Due to the available
    implementation this has to be done per-wiki, so is perhaps slower than
    necessary. For current dataset sizes that isn't an issue.

    We need to bucketize the number of sessions so we can split the
    per-wiki input into strata and sample from each bucket. This helps
    ensure we keep a consistent distribution of popular and unpopular
    queries from the input to the output.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input dataframe to have bucketing fractions calculated for.
    num_buckets : int
        Number of buckets to create per-wiki

    Returns
    -------
    list of ints
        List of right-edges of buckets that will have an approximately equal number
        of queries per bucket.
    """

    percentiles = [x/float(num_buckets) for x in range(1, num_buckets)]
    # With 100 buckets, there will be buckets at .01, .02, etc. This specifies
    # percentils .01 must be the value between .009 and .011
    relative_error = 1. / (num_buckets * 10)
    splits = df.approxQuantile('num_sessions', percentiles, relative_error)

    # range(1, num_buckets) returned num_buckets-1 values. This final inf captures
    # everything from the last bucket to the end.
    return splits + [float('inf')]


def _sample_queries(df, wikis, num_buckets=100, samples_desired=10000, seed=None):
    """Sample down a unique list of (wiki, norm_query, num_sessions)

    Given a dataset of unique queries, sample it down to samples_desired per wiki
    maintaining the distribution of queries with many sessions and queries with
    few sessions.

    Has a few drawbacks in that data is serialized from java to python and back
    multiple times. Not sure how to address that yet. Would be nice if we could
    use pyspark.ml.feature.Bucketizer, but that only takes a single column to
    bucketize so we would have to apply to an RDD per wiki, unioning them
    together at the end.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input dataframe containing (wiki, norm_query, num_sessions) fields.
    wikis : list of strings
        List of wikis to generate samples for.
    num_buckets : int, optional
        The number of buckets to divide each wiki's queries into. An equal number
        of queries will be sampled from each bucket. (Default: 100)
    samples_desired : int, optional
        The approximate total number of samples to return per wiki.
        (Default: 10000)
    seed : int or None, optional
        Seed used for random sampling. (Default: None)

    Returns
    -------
    pyspark.sql.DataFrame
        The set of sampled (wikiid, norm_query) rows desired with approximately
        samples_desired rows per wikiid.
    """

    # Number of samples we need per bucket
    bucket_samples = samples_desired / num_buckets
    # Map from wikiid -> list of splits, used by find_split to bucketize
    wiki_splits = {}
    # Map from (wikiid, split) -> % of samples needed. Used by RDD.sampleByKey
    wiki_fractions = {}
    for wiki in wikis:
        df_wiki = df.where(df.wikiid == wiki).cache()
        try:
            num_rows = df_wiki.count()
            # If we have less than the desired amount of data no sampling is needed
            if num_rows < samples_desired:
                wiki_fractions[(wiki, float('inf'))] = 1.
                wiki_splits[wiki] = [float('inf')]
                continue

            # Number of source rows expected in each bucket
            bucket_rows = float(num_rows) / num_buckets
            # Fraction of rows needed from each bucket
            bucket_fraction = bucket_samples / bucket_rows
            wiki_splits[wiki] = _calc_splits(df_wiki, num_buckets)
            for split in wiki_splits[wiki]:
                wiki_fractions[(wiki, split)] = bucket_fraction
        finally:
            df_wiki.unpersist()

    def to_pair_rdd(row):
        splits = wiki_splits[row.wikiid]
        # Find leftmost split greater than or equal to row.num_sessions
        idx = bisect.bisect_left(splits, row.num_sessions)
        if idx == len(splits):
            raise ValueError
        split = splits[idx]
        return ((row.wikiid, split), (row.wikiid, row.norm_query))

    return (
        df.rdd
        # To use sampleByKey we need to convert to a PairRDD which has keys matching
        # those used in wiki_fractions.
        .map(to_pair_rdd)
        .sampleByKey(withReplacement=False, fractions=wiki_fractions, seed=seed)
        # Convert the PairRDD back into a dataframe.
        .map(lambda (key, row): row)
        .toDF(['wikiid', 'norm_query']))


def sample(df, wikis, seed=None, queries_per_wiki=10000,
           min_sessions_per_query=35, max_queries_per_ip_day=50):
    """Choose a representative sample of queries from input dataframe.

    Takes in the unsampled query click logs and filters it down into a smaller
    representative sample that can be run through the machine learning
    pipeline. Note that when using data in the `discovery.query_clicks_daily`
    table the query needs to be post-processed to normalize the queries for
    grouping.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input dataframe with columns wikiid, query, session_id and q_by_ip_day.
    wikis : set of strings
        The set of wikis to sample for. Many wikis will not have enough data
        to generate reasonable ml models. TODO: Should we instead define a
        minimum size to require?
    seed : int or None, optional
        The random seed used when sampling. If None a seed will be chosen
        randomly. (Default: None)
    queries_per_wiki : int, optional
        The desired number of distinct normalized queries per wikiid in the
        output. This constraint is approximate and the returned number
        of queries may slightly vary per wiki. (Default: 10000)
    min_sessions_per_query : int, optional
        Require each chosen query to have at least this many sessions per
        query. This is necessary To train the DBN later in the pipeline.
        (Default: 35)
    max_queries_per_ip_day : int, optional
        Requires each chosen query to have at most this many full text searches
        issued from it's IP on the day the query was issued. This Filters out
        high volume users which are quite possibly bots or other non-standard
        sessions. (Default: 50)

    Returns
    -------
    pyspark.sql.DataFrame
        The input DataFrame with all columns it origionally had sampled down
        based on the provided constraints.
    """
    mjolnir.spark.assert_columns(df, ['wikiid', 'norm_query', 'session_id', 'q_by_ip_day'])

    # Filter down the input into the wikis we care about and remove sessions
    # from overly active users, which are presumably bots.
    df_filtered = (
        df
        .where(_array_contains(F.array([F.lit(wiki) for wiki in wikis]),
                               F.col('wikiid')))
        .where(df.q_by_ip_day <= max_queries_per_ip_day)
        .drop(df.q_by_ip_day))

    # Aggregate down into a unique set of (wikiid, norm_query) and add in a
    # count of the number of unique sessions per pair. Filter on the number
    # of sessions as we need some minimum number of sessions per query to train
    # the DBN
    df_queries_unique = (
        df_filtered
        .groupBy('wikiid', 'norm_query')
        # To make QuantileDiscretizer happy later on, we need
        # to cast this to a double. Can be removed in 2.x which
        # accepts anything numeric.
        .agg(F.countDistinct('session_id').cast('double').alias('num_sessions'))
        .where(F.col('num_sessions') >= min_sessions_per_query)
        # This rdd will be used multiple times through strata generation and
        # sampling. Cache to not duplicate the filtering and aggregation work.
        # Spark will eventually throw this away in an LRU fashion.
        .cache())

    df_queries_sampled = _sample_queries(df_queries_unique, wikis, samples_desired=queries_per_wiki, seed=seed)

    # Select the rows chosen by sampling from the filtered df
    return df_filtered.join(df_queries_sampled, how='inner', on=['wikiid', 'norm_query'])
