from __future__ import absolute_import
import itertools
import math
import mjolnir.sampling
import numpy as np
from pyspark.sql import functions as F
import string


def test_sampling_selects_all_if_less_than_samples_per_wiki(spark_context, hive_context):
    # Although hive_context is not used directly, it must be requested so it
    # can monkey-patch pyspark.RDD

    # Generate a sample dataframe with only 10 queries, then ask for 10000
    df = spark_context.parallelize([
        ('foo', 'a', 1, 'aaa', list(range(3))),
        ('foo', 'b', 2, 'ccc', list(range(3))),
        ('foo', 'c', 3, 'ccc', list(range(3))),
        ('foo', 'd', 4, 'ddd', list(range(3))),
        ('foo', 'e', 5, 'eee', list(range(3))),
    ]).toDF(['wikiid', 'query', 'norm_query_id', 'session_id', 'hit_page_ids'])

    hit_page_id_counts, df_sampled = mjolnir.sampling.sample(
        df, samples_per_wiki=100, seed=12345)
    sampled = df_sampled.collect()
    # The sampling rate should have been chosen as 1.0, so we should have all data
    # regardless of probabilities.
    assert len(sampled) == 5


def test_sampling_general_approach(spark_context, hive_context):
    """Generate a dataframe and see if sampling it has same general shape"""

    # Create wikis with different looking long tail distributions
    wikis = [
        ("foowiki", 1500, -1),
        ("barwiki", 700, -1),
        # This has a very flat long tail, with most data points being the same
        ("bazwiki", 700, -2),
    ]
    # use all combinations of lowercase letters as our set of test queries. This is 26^2,
    # or just shy of 700 queries.
    queries = ["%s%s" % pair for pair in itertools.product(string.ascii_lowercase, string.ascii_lowercase)]
    rows = []
    for (wiki, a, k) in wikis:
        # create sessions for each query with a long tail distribution
        for (x, q) in enumerate(queries):
            # approximate a long tail distribution using ax^k + b
            # x + 1 needed because enumerate starts at 0. b is set to 10 to test the
            # min sessions per query limit
            num_sessions = max(1, min(100, int(a * math.pow(x+1, k)) + 10))
            for j in xrange(0, num_sessions):
                session_id = "%s_%s_%s" % (wiki, q, str(j))
                rows.append((wiki, q, x, session_id, list(range(3))))

    df = (
        spark_context.parallelize(rows)
        .toDF(['wikiid', 'query', 'norm_query_id', 'session_id', 'hit_page_ids']))

    samples_per_wiki = 1000
    # Using a constant seed ensures deterministic testing. Because this code
    # actually relies on the law of large numbers, and we do not have large
    # numbers here, many seeds probably fail.
    hit_page_id_counts, df_sampled = mjolnir.sampling.sample(
        df, samples_per_wiki=samples_per_wiki, seed=12345)
    sampled = (
        df_sampled
        .select('wikiid', 'query', F.explode('hit_page_ids').alias('hit_page_id'))
        .drop_duplicates()
        .groupBy('wikiid')
        .agg(F.count(F.lit(1)).alias('num_samples'))
        .collect())

    total_samples_desired = len(wikis) * samples_per_wiki
    total_samples = sum([r.num_samples for r in sampled])
    assert abs(total_samples - total_samples_desired) / float(total_samples_desired) < 0.05
    # Test each wiki also meets the constraint
    for (wiki, _, _) in wikis:
        # ratio of rows
        sampled_num_rows = sum([r.num_samples for r in sampled if r.wikiid == wiki])
        assert abs(sampled_num_rows - samples_per_wiki) / float(samples_per_wiki) < 0.05

    # assert correlation between sessions per query
    orig_grouped = (
        df.groupBy('wikiid', 'norm_query_id')
        .agg(F.countDistinct('session_id').alias('num_sessions'))
        .collect())
    sampled_grouped = (
        df_sampled.groupBy('wikiid', 'norm_query_id')
        .agg(F.countDistinct('session_id').alias('num_sessions'))
        .collect())

    for (wiki, _, _) in wikis:
        orig = sorted([r.num_sessions for r in orig_grouped if r.wikiid == wiki])
        sampled = sorted([r.num_sessions for r in sampled_grouped if r.wikiid == wiki])
        # interpolate sampled into the same length as orig
        sampled_interp = np.interp(range(len(orig)),
                                   np.linspace(1, len(orig), len(sampled)),
                                   sampled)
        # corrcoef allows comparing N data sets, returning a covariance matrix.
        # take 0,1 to get corr(orig, sampled_interp)
        corr = np.corrcoef(orig, sampled_interp)[0, 1]
        # Is .8 reasonable? Sometimes this fails when using something stricter
        # like .95
        assert corr > .8, wiki
