"""
Takes in the full click logs and samples down to a set appropriate
for running through the MLR pipeline.

Is perhaps overly complex because the utilities in spark aren't
particularly designed to work with multiple datasets (wikis)
in a single dataframe, but it works well enough for our data sizes.
"""

from __future__ import absolute_import
import bisect
import mjolnir.spark
from pyspark.sql import functions as F


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


def _sample_queries(df, wiki_percents, num_buckets=100, seed=None):
    """Sample down a unique list of (wiki, norm_query_id, num_sessions)

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
        Input dataframe containing (wiki, norm_query_id, num_sessions) fields.
    wiki_percents : dict
        Map from wikiid to the fraction of norm_query_ids to use from that wiki.
    num_buckets : int, optional
        The number of buckets to divide each wiki's queries into. An equal number
        of queries will be sampled from each bucket. (Default: 100)
    seed : int or None, optional
        Seed used for random sampling. (Default: None)

    Returns
    -------
    pyspark.sql.DataFrame
        The set of sampled (wikiid, norm_query_id) rows desired with approximately
        samples_desired rows per wikiid.
    """

    # Map from wikiid -> list of splits, used by find_split to bucketize
    wiki_splits = {}
    # Map from (wikiid, split) -> % of samples needed. Used by RDD.sampleByKey
    wiki_fractions = {}
    for wiki, fraction in wiki_percents.items():
        # If we have less than the desired amount of data no sampling is needed
        if fraction >= 1.:
            wiki_fractions[(wiki, float('inf'))] = 1.
            wiki_splits[wiki] = [float('inf')]
            continue

        df_wiki = df.where(df.wikiid == wiki)
        wiki_splits[wiki] = _calc_splits(df_wiki, num_buckets)
        for split in wiki_splits[wiki]:
            wiki_fractions[(wiki, split)] = fraction

    def to_pair_rdd(row):
        splits = wiki_splits[row.wikiid]
        # Find leftmost split greater than or equal to row.num_sessions
        idx = bisect.bisect_left(splits, row.num_sessions)
        if idx == len(splits):
            raise ValueError
        split = splits[idx]
        return ((row.wikiid, split), (row.wikiid, row.norm_query_id))

    return (
        df.rdd
        # To use sampleByKey we need to convert to a PairRDD which has keys matching
        # those used in wiki_fractions.
        .map(to_pair_rdd)
        .sampleByKey(withReplacement=False, fractions=wiki_fractions, seed=seed)
        # Convert the PairRDD back into a dataframe.
        .map(lambda (key, row): row)
        .toDF(['wikiid', 'norm_query_id']))


def sample(df, seed=None, samples_per_wiki=1000000):
    """Choose a representative sample of queries from input dataframe.

    Takes in the unsampled query click logs and filters it down into a smaller
    representative sample that can be run through the machine learning
    pipeline. Note that when using data in the `discovery.query_clicks_daily`
    table the query needs to be post-processed to normalize the queries for
    grouping.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input dataframe with columns wikiid, query, and session_id.
    seed : int or None, optional
        The random seed used when sampling. If None a seed will be chosen
        randomly. (Default: None)
    samples_per_wiki : int, optional
        The desired number of distinct (query, hit_page_id) pairs in the
        output. This constraint is approximate and the returned number
        of queries may vary per wiki. (Default: 1000000)

    Returns
    -------
    pyspark.sql.DataFrame
        The input DataFrame with all columns it origionally had sampled down
        based on the provided constraints.
    """
    mjolnir.spark.assert_columns(df, ['wikiid', 'query', 'hit_page_ids', 'norm_query_id', 'session_id'])

    # We need this df twice, and by default the df coming in here is from
    # mjolnir.norm_query which is quite expensive.
    df.cache()

    # Figure out the percentage of each wiki's norm_query_id's we need to approximately
    # have samples_per_wiki final training samples.
    hit_page_id_counts = (
        df
        .select('wikiid', 'query', F.explode('hit_page_ids').alias('hit_page_id'))
        # We could groupBy('wikiid').agg(F.countDistinct('query', 'hit_page_id'))
        # directly, but this causes spark to blow out memory limits by
        # collecting too much data on too few executors.
        .groupBy('wikiid', 'query')
        .agg(F.countDistinct('hit_page_id').alias('num_hit_page_ids'))
        .groupBy('wikiid')
        .agg(F.sum('num_hit_page_ids').alias('num_hit_page_ids'))
        .collect())

    hit_page_id_counts = {row.wikiid: row.num_hit_page_ids for row in hit_page_id_counts}

    wiki_percents = {}
    needs_sampling = False

    for wikiid, num_hit_page_ids in hit_page_id_counts.items():
        wiki_percents[wikiid] = min(1., float(samples_per_wiki) / num_hit_page_ids)
        if wiki_percents[wikiid] < 1.:
            needs_sampling = True

    if not needs_sampling:
        return hit_page_id_counts, df

    # Aggregate down into a unique set of (wikiid, norm_query_id) and add in a
    # count of the number of unique sessions per pair. We will sample per-strata
    # based on percentiles of num_sessions.
    df_queries_unique = (
        df
        .groupBy('wikiid', 'norm_query_id')
        .agg(F.countDistinct('session_id').alias('num_sessions'))
        # This rdd will be used multiple times through strata generation and
        # sampling. Cache to not duplicate the filtering and aggregation work.
        .cache())

    df_queries_sampled = _sample_queries(df_queries_unique, wiki_percents, seed=seed)

    # Select the rows chosen by sampling from the input df
    df_sampled = (
        df
        .join(df_queries_sampled, how='inner', on=['wikiid', 'norm_query_id'])
        .cache())

    return hit_page_id_counts, df_sampled
