"""
Transforms user behaviour logs into a labeled dataset
"""

from __future__ import absolute_import
import argparse
import datetime
import logging
import mjolnir.dbn
import mjolnir.kafka.client
import mjolnir.metrics
import mjolnir.norm_query
import mjolnir.features
import mjolnir.sampling
from pyspark import SparkContext
from pyspark.sql import HiveContext
from pyspark.sql import functions as F
import requests


def run_pipeline(sc, sqlContext, input_dir, output_dir, wikis, samples_per_wiki,
                 min_sessions_per_query, search_cluster, brokers,
                 samples_size_tolerance, kafka_request_topic, kafka_result_topic,
                 session_factory=requests.Session):

    if brokers:
        brokers = mjolnir.kafka.client.ClientConfig(
            brokers, kafka_request_topic, kafka_result_topic)

    sqlContext.sql("DROP TEMPORARY FUNCTION IF EXISTS stemmer")
    sqlContext.sql("CREATE TEMPORARY FUNCTION stemmer AS 'org.wikimedia.analytics.refinery.hive.StemmerUDF'")

    # Load click data from HDFS
    df_clicks = (
        sqlContext.read.parquet(input_dir)
        # Limit to the wikis we are working against
        .where(F.col('wikiid').isin(wikis))
        # Drop requests from 'too busy' IP's. These are plausibly bots, or maybe just proxys.
        .where(F.col('q_by_ip_day') < 50)
        .drop('q_by_ip_day')
        # Clicks and hits contains a bunch of useful debugging data, but we don't
        # need any of that here. Save a bunch of memory by only working with
        # lists of page ids
        .withColumn('hit_page_ids', F.col('hits.pageid'))
        .drop('hits')
        .withColumn('click_page_ids', F.col('clicks.pageid'))
        .drop('clicks'))

    # Normalize queries into groups of related queries. This helps to have a larger
    # number of sessions per normalized query to train the DBN on.
    # Note that df_norm comes back cached
    df_norm = mjolnir.norm_query.transform(
        df_clicks,
        url_list=mjolnir.cirrus.SEARCH_CLUSTERS[search_cluster] if search_cluster else None,
        brokers=brokers,
        # TODO: While this works for now, at some point we might want to handle
        # things like multimedia search from commons, and non-main namespace searches.
        indices={wiki: '%s_content' % (wiki) for wiki in wikis},
        min_sessions_per_query=min_sessions_per_query,
        session_factory=session_factory)

    # Sample to some subset of queries per wiki
    hit_page_id_counts, df_sampled_raw = mjolnir.sampling.sample(
        df_norm,
        seed=54321,
        samples_per_wiki=samples_per_wiki)

    # Transform our dataframe into the shape expected by the DBN
    df_sampled = (
        df_sampled_raw
        # Explode source into a row per displayed hit
        .select('*', F.expr("posexplode(hit_page_ids)").alias('hit_position', 'hit_page_id'))
        .drop('hit_page_ids')
        # Mark all hits that were clicked by a user
        .withColumn('clicked', F.expr('array_contains(click_page_ids, hit_page_id)'))
        .drop('click_page_ids')
        .cache())

    # Learn relevances
    df_rel = (
        mjolnir.dbn.train(df_sampled, dbn_config={
            'MAX_ITERATIONS': 40,
            'MIN_DOCS_PER_QUERY': 10,
            'MAX_DOCS_PER_QUERY': 20,
            'DEFAULT_REL': 0.5,
            'GAMMA': 0.9})
        # naive conversion of relevance % into a label
        .withColumn('label', (F.col('relevance') * 10).cast('int')))

    # Merge relevance back into our sampled dataset. This join has to happen
    # before aggregation for the weighted ndcg to be calculated with the
    # original result ordering.
    df_all_hits = (
        df_sampled
        .select('wikiid', 'query', 'norm_query_id', 'hit_page_id', 'session_id', 'hit_position')
        .join(df_rel, how='inner', on=['wikiid', 'norm_query_id', 'hit_page_id'])
        .cache())

    # Aggregate per-session rows into per-page rows
    df_hits = (
        df_all_hits
        .groupBy('wikiid', 'query', 'norm_query_id', 'hit_page_id')
        # weight is now the number of times a hit was displayed to a user
        .agg(F.count(F.lit(1)).alias('weight'),
             F.mean('hit_position').alias('hit_position'),
             # These should be the same per group, but to keep things easy
             # take first rather than grouping
             F.first('label').alias('label'),
             F.first('relevance').alias('relevance'))
        .cache())

    # Check that we really got enough stuff
    actual_samples_per_wiki = df_hits.groupby('wikiid').agg(F.count(F.lit(1)).alias('n_obs')).collect()
    actual_samples_per_wiki = {row.wikiid: row.n_obs for row in actual_samples_per_wiki}

    not_enough_samples = []
    for wiki in wikis:
        # We cant have more samples than we started with
        expected = min(samples_per_wiki, hit_page_id_counts[wiki])
        try:
            actual = actual_samples_per_wiki[wiki]
        except KeyError:
            # This will probably still error, but give better messages.
            actual = 0
        if actual == 0 or expected / float(actual) < samples_size_tolerance:
            not_enough_samples.append(
                'Collected %d samples from %s which is less than %d%% of the requested sample size %d'
                % (actual, wiki, samples_size_tolerance*100, expected))
    if not_enough_samples:
        raise ValueError('\n'.join(not_enough_samples))

    print('Fetched a total of %d samples for %d wikis' % (sum(actual_samples_per_wiki.values()), len(wikis)))

    # Calculate a few stats about the dataset
    weightedNdcgAt10 = mjolnir.metrics.ndcg(df_all_hits, 10, query_cols=['wikiid', 'query', 'session_id'])
    print('weighted ndcg@10:')
    for wiki, ndcg in weightedNdcgAt10.items():
        print('\t%s: %.4f' % (wiki, ndcg))

    ndcgAt10 = mjolnir.metrics.ndcg(df_hits, 10, query_cols=['wikiid', 'query'])
    print('unweighted ndcg@10:')
    for wiki, ndcg in ndcgAt10.items():
        print('\t%s: %.4f' % (wiki, ndcg))

    # Add some metadata and write it all out
    (
        df_hits
        .withColumn('label', mjolnir.spark.add_meta(sc, F.col('label'), {
            'click_log_weighted_ndcg@10': weightedNdcgAt10,
            'click_log_ndcg@10': ndcgAt10,
            'collected_at': datetime.datetime.now().isoformat(),
            'used_kafka': brokers is not None,
            'search_cluster': search_cluster if search_cluster else "",
            # TODO: Where does this metadata go? It seems a bit more top-level
            # but could be useful to remember.
            'min_sessions_per_query': min_sessions_per_query,
            'input_dir': input_dir,
            'output_dir': output_dir,
        }))
        .write.parquet(output_dir))


def percentage(value):
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be between 0 and 1")
    if value < 0 or value > 1:
        raise argparse.ArgumentTypeError("Must be between 0 and 1")
    return value


def arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-i', '--input', dest='input_dir', type=str,
        default='hdfs://analytics-hadoop/wmf/data/discovery/query_clicks/daily/year=*/month=*/day=*',
        help='Input path, prefixed with hdfs://, to query and click data')
    parser.add_argument(
        '-q', '--samples-per-wiki', dest='samples_per_wiki', type=int, default=1000000,
        help='The approximate number of rows in the final result per-wiki.')
    parser.add_argument(
        '-qe', '--sample-size-tolerance', dest='samples_size_tolerance', type=percentage, default=0.5,
        help='The tolerance between the --samples-per-wiki set and the actual number of rows fetched.'
             + ' Higher requires closer match.')
    parser.add_argument(
        '-s', '--min-sessions', dest='min_sessions_per_query', type=int, default=10,
        help='The minimum number of sessions per normalized query')
    parser.add_argument(
        '-c', '--search-cluster', dest='search_cluster', type=str,
        choices=mjolnir.cirrus.SEARCH_CLUSTERS.keys(), help='Search cluster to source features from')
    parser.add_argument(
        '-o', '--output-dir', dest='output_dir', type=str, required=True,
        help='Output path, prefixed with hdfs://, to write resulting dataframe to')
    parser.add_argument(
        '-k', '--kafka', metavar='BROKER', dest='brokers', type=str, nargs='+',
        help='Collect feature vectors via kafka using specified broker in <host>:<port> '
             + ' form to bootstrap access. Query normalization will still use the '
             + ' --search-cluster option')
    parser.add_argument(
        '--kafka-request-topic', metavar='TOPIC', dest='kafka_request_topic',
        default=mjolnir.kafka.TOPIC_REQUEST, type=str,
        help='TODO')
    parser.add_argument(
        '--kafka-result-topic', metavar='TOPIC', dest='kafka_result_topic',
        default=mjolnir.kafka.TOPIC_RESULT, type=str,
        help='TODO')
    parser.add_argument(
        'wikis', metavar='wiki', type=str, nargs='+',
        help='A wiki to generate labels for')
    return parser


def main(**kwargs):
    if len(kwargs['brokers']) == 1 and ',' in kwargs['brokers'][0]:
        kwargs['brokers'] = kwargs['brokers'][0].split(',')

    sc = SparkContext(appName="MLR: data collection pipeline")
    # spark info logging is incredibly spammy. Use warn to have some hope of
    # human decipherable output
    sc.setLogLevel('WARN')
    sqlContext = HiveContext(sc)
    run_pipeline(sc, sqlContext, **kwargs)


if __name__ == "__main__":
    logging.basicConfig()
    kwargs = dict(vars(arg_parser().parse_args()))
    main(**kwargs)
