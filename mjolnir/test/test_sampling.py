import itertools
import math
import mjolnir.sampling
import numpy as np
from pyspark.sql import functions as F
import string


def test_sampling_selects_all_if_less_than_queries_per_wiki(spark_context, hive_context):
    # Although hive_context is not used directly, it must be requested so it
    # can monkey-patch pyspark.RDD

    # Generate a sample dataframe with only 10 queries, then ask for 10000
    df = spark_context.parallelize([
        ('foo', 'a', 'aaa', 1),
        ('foo', 'b', 'ccc', 2),
        ('foo', 'c', 'ccc', 2),
        ('foo', 'd', 'ddd', 2),
        ('foo', 'e', 'eee', 2),
    ]).toDF(['wikiid', 'norm_query', 'session_id', 'q_by_ip_day'])

    sampled = mjolnir.sampling.sample(df, ['foo'], queries_per_wiki=100,
                                      min_sessions_per_query=1, seed=12345).collect()
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
                rows.append((wiki, q, session_id, 1))

    df = spark_context.parallelize(rows).toDF(['wikiid', 'norm_query', 'session_id', 'q_by_ip_day'])
    queries_per_wiki = 100
    df_sampled = mjolnir.sampling.sample(df, [wiki for (wiki, _, _) in wikis],
                                         queries_per_wiki=queries_per_wiki,
                                         min_sessions_per_query=10, seed=12345)
    sampled = df_sampled.collect()

    ratio_of_sessions = len(sampled) / len(rows)
    expected_ratio_of_sessions = queries_per_wiki / len(queries)
    # assert the overall sampling matches constraint on ratio
    assert abs(ratio_of_sessions - expected_ratio_of_sessions) < 0.01
    # Test each wiki also meets the constraint
    for (wiki, _, _) in wikis:
        # ratio of rows
        sampled_num_rows = len([r for r in sampled if r.wikiid == wiki])
        orig_num_rows = len([r for r in rows if r[0] == wiki])
        ratio_of_sessions = sampled_num_rows / orig_num_rows
        assert abs(ratio_of_sessions - expected_ratio_of_sessions) < 0.01, wiki

    # assert correlation between sessions per query
    orig_grouped = (
        df.groupBy('wikiid', 'norm_query')
        .agg(F.countDistinct('session_id').alias('num_sessions'))
        .collect())
    sampled_grouped = (
        df_sampled.groupBy('wikiid', 'norm_query')
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
