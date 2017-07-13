"""
Example script demonstrating the data collection portion of the MLR pipeline.
This is mostly to demonstrate how everything ties together

To run:
    PYSPARK_PYTHON=venv/bin/python spark-submit \
        --jars /path/to/mjolnir-with-dependencies.jar
        --artifacts 'mjolnir_venv.zip#venv' \
        --files /usr/lib/libhdfs.so.0.0.0
        mjolnir/cli/data_pipeline.py
"""

import argparse
import mjolnir.dbn
import mjolnir.metrics
import mjolnir.norm_query
import mjolnir.features
import mjolnir.sampling
from pyspark import SparkContext
from pyspark.sql import HiveContext
from pyspark.sql import functions as F

SEARCH_CLUSTERS = {
    'eqiad': ['http://elastic%d.eqiad.wmnet:9200/_msearch' % (i) for i in range(1017, 1052)],
    'codfw': ['http://elastic%d.codfw.wmnet:9200/_msearch' % (i) for i in range(2001, 2035)],
}


def main(sc, sqlContext, input_dir, output_dir, wikis, queries_per_wiki,
         min_sessions_per_query, search_cluster, brokers):
    # TODO: Should this jar have to be provided on the command line instead?
    sqlContext.sql("ADD JAR /mnt/hdfs/wmf/refinery/current/artifacts/refinery-hive.jar")
    sqlContext.sql("CREATE TEMPORARY FUNCTION stemmer AS 'org.wikimedia.analytics.refinery.hive.StemmerUDF'")

    # Load click data from HDFS
    df_clicks = (
        sqlContext.read.parquet(input_dir)
        # Limit to the wikis we are working against
        .where(mjolnir.sampling._array_contains(F.array(map(F.lit, wikis)), F.col('wikiid')))
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
        url_list=SEARCH_CLUSTERS[search_cluster],
        # TODO: While this works for now, at some point we might want to handle
        # things like multimedia search from commons, and non-main namespace searches.
        indices={wiki: '%s_content' % (wiki) for wiki in wikis},
        min_sessions_per_query=min_sessions_per_query)

    # Sample to some subset of queries per wiki
    df_sampled = (
        mjolnir.sampling.sample(
            df_norm,
            wikis=wikis,
            seed=54321,
            queries_per_wiki=queries_per_wiki,
            min_sessions_per_query=min_sessions_per_query)
        # Explode source into a row per displayed hit
        .select('*', F.expr("posexplode(hit_page_ids)").alias('hit_position', 'hit_page_id'))
        .drop('hit_page_ids')
        # Mark all hits that were clicked by a user
        .withColumn('clicked', F.expr('array_contains(click_page_ids, hit_page_id)'))
        .drop('click_page_ids')
        .cache())

    # materialize df_sampled and unpersist df_norm
    df_sampled.count()
    df_norm.unpersist()

    # Learn relevances
    df_rel = (
        mjolnir.dbn.train(df_sampled, {
            'MAX_ITERATIONS': 40,
            'DEBUG': False,
            'PRETTY_LOG': True,
            'MIN_DOCS_PER_QUERY': 10,
            'MAX_DOCS_PER_QUERY': 20,
            'SERP_SIZE': 20,
            'QUERY_INDEPENDENT_PAGER': False,
            'DEFAULT_REL': 0.5})
        # naive conversion of relevance % into a label
        .withColumn('label', (F.col('relevance') * 10).cast('int')))

    df_all_hits = (
        df_sampled
        .select('wikiid', 'query', 'norm_query_id', 'hit_page_id', 'session_id', 'hit_position')
        .join(df_rel, how='inner', on=['wikiid', 'norm_query_id', 'hit_page_id'])
        .cache())

    # materialize df_all_hits and drop df_sampled, df_norm
    df_all_hits.count()
    df_sampled.unpersist()

    # TODO: Training is per-wiki, should this be as well?
    weightedNdcgAt10 = mjolnir.metrics.ndcg(df_all_hits, 10, query_cols=['wikiid', 'query', 'session_id'])
    print 'weighted ndcg@10: %.4f' % (weightedNdcgAt10)

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

    # materialize df_hits and drop df_all_hits
    df_hits.count()
    df_all_hits.unpersist()

    # TODO: Training is per-wiki, should this be as well?
    ndcgAt10 = mjolnir.metrics.ndcg(df_hits, 10, query_cols=['wikiid', 'query'])
    print 'unweighted ndcg@10: %.4f' % (ndcgAt10)

    # Collect features for all known queries. Note that this intentionally
    # uses query and NOT norm_query_id. Merge those back into the source hits.
    if brokers:
        df_features = mjolnir.features.collect_kafka(
            df_hits,
            brokers=brokers,
            indices={wiki: '%s_content' % (wiki) for wiki in wikis},
            feature_definitions=mjolnir.features.enwiki_features())
    else:
        df_features = mjolnir.features.collect_es(
            df_hits,
            url_list=SEARCH_CLUSTERS[search_cluster],
            # TODO: While this works for now, at some point we might want to handle
            # things like multimedia search from commons, and non-main namespace searches.
            indices={wiki: '%s_content' % (wiki) for wiki in wikis},
            # TODO: If we are going to do multiple wikis, this probably needs different features
            # per wiki? At a minimum trying to include useful templates as features will need
            # to vary per-wiki. Varied features per wiki would also mean they can't be trained
            # together, which is perhaps a good thing anyways.
            feature_definitions=mjolnir.features.enwiki_features())
    df_hits_with_features = (
        df_hits
        .join(df_features, how='inner', on=['wikiid', 'query', 'hit_page_id'])
        .withColumn('label', mjolnir.spark.add_meta(sc, F.col('label'), {
            'weightedNdcgAt10': weightedNdcgAt10,
            'ndcgAt10': ndcgAt10,
        })))

    df_hits_with_features.write.parquet(output_dir)


def parse_arguments():
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument(
        '-i', '--input', dest='input_dir', type=str,
        default='hdfs://analytics-hadoop/wmf/data/discovery/query_clicks/daily/year=*/month=*/day=*',
        help='Input path, prefixed with hdfs://, to query and click data')
    parser.add_argument(
        '-q', '--queries-per-wiki', dest='queries_per_wiki', type=int, default=20000,
        help='The number of normalized queries, per wiki, to operate on')
    parser.add_argument(
        '-s', '--min-sessions', dest='min_sessions_per_query', type=int, default=10,
        help='The minimum number of sessions per normalized query')
    parser.add_argument(
        '-c', '--search-cluster', dest='search_cluster', type=str, default='codfw',
        choices=SEARCH_CLUSTERS.keys(), help='Search cluster to source features from')
    parser.add_argument(
        '-o', '--output-dir', dest='output_dir', type=str, required=True,
        help='Output path, prefixed with hdfs://, to write resulting dataframe to')
    parser.add_argument(
        '-k', '--kafka', metavar='BROKER', dest='brokers', type=str, nargs='+',
        help='Collect feature vectors via kafka using specified broker in <host>:<port> '
             + ' form to bootstrap access. Query normalization will still use the '
             + ' --search-cluster option')
    parser.add_argument(
        'wikis', metavar='wiki', type=str, nargs='+',
        help='A wiki to generate features and labels for')

    args = parser.parse_args()
    return dict(vars(args))


if __name__ == "__main__":
    args = parse_arguments()
    sc = SparkContext(appName="MLR: data collection pipeline")
    # spark info logging is incredibly spammy. Use warn to have some hope of
    # human decipherable output
    sc.setLogLevel('WARN')
    sqlContext = HiveContext(sc)
    main(sc, sqlContext, **args)
