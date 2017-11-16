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

from __future__ import absolute_import
import argparse
from collections import OrderedDict
import logging
import mjolnir.dbn
import mjolnir.metrics
import mjolnir.norm_query
import mjolnir.features
import mjolnir.sampling
from pyspark import SparkContext
from pyspark.sql import HiveContext
from pyspark.sql import functions as F

SEARCH_CLUSTERS = {
    'eqiad': ['http://elastic%d.eqiad.wmnet:9200' % (i) for i in range(1017, 1052)],
    'codfw': ['http://elastic%d.codfw.wmnet:9200' % (i) for i in range(2001, 2035)],
}


def run_pipeline(sc, sqlContext, input_dir, output_dir, wikis, samples_per_wiki,
                 min_sessions_per_query, search_cluster, brokers, ltr_feature_definitions,
                 samples_size_tolerance):
    # TODO: Should this jar have to be provided on the command line instead?
    sqlContext.sql("ADD JAR /mnt/hdfs/wmf/refinery/current/artifacts/refinery-hive.jar")
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
        url_list=SEARCH_CLUSTERS[search_cluster],
        # TODO: While this works for now, at some point we might want to handle
        # things like multimedia search from commons, and non-main namespace searches.
        indices={wiki: '%s_content' % (wiki) for wiki in wikis},
        min_sessions_per_query=min_sessions_per_query)

    # Sample to some subset of queries per wiki
    df_sampled = (
        mjolnir.sampling.sample(
            df_norm,
            seed=54321,
            samples_per_wiki=samples_per_wiki)
        # Explode source into a row per displayed hit
        .select('*', F.expr("posexplode(hit_page_ids)").alias('hit_position', 'hit_page_id'))
        .drop('hit_page_ids')
        # Mark all hits that were clicked by a user
        .withColumn('clicked', F.expr('array_contains(click_page_ids, hit_page_id)'))
        .drop('click_page_ids')
        .cache())

    # materialize df_sampled and unpersist df_norm
    nb_samples = df_sampled.count()
    if ((nb_samples / float(len(wikis)*samples_per_wiki)) < samples_size_tolerance):
        raise ValueError('Collected %d samples this is less than %d%% of the requested sample size %d'
                         % (nb_samples, samples_size_tolerance*100, samples_per_wiki))
    print 'Fetched a total of %d samples for %d wikis' % (nb_samples, len(wikis))
    df_norm.unpersist()

    # Target around 125k rows per partition. Note that this isn't
    # how many the dbn will see, because it gets collected up. Just
    # a rough guess.
    dbn_partitions = int(max(200, min(2000, nb_samples / 125000)))

    # Learn relevances
    df_rel = (
        mjolnir.dbn.train(df_sampled, num_partitions=dbn_partitions, dbn_config={
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
    fnames_accu = df_hits._sc.accumulator(OrderedDict(), mjolnir.features.FeatureNamesAccumulator())
    if ltr_feature_definitions:
        if brokers:
            df_features = mjolnir.features.collect_from_ltr_plugin_and_kafka(
                df_hits,
                brokers=brokers,
                indices={wiki: '%s_content' % (wiki) for wiki in wikis},
                model=ltr_feature_definitions,
                feature_names_accu=fnames_accu)
        else:
            df_features = mjolnir.features.collect_from_ltr_plugin(
                df_hits,
                url_list=SEARCH_CLUSTERS[search_cluster],
                # TODO: While this works for now, at some point we might want to handle
                # things like multimedia search from commons, and non-main namespace searches.
                indices={wiki: '%s_content' % (wiki) for wiki in wikis},
                model=ltr_feature_definitions,
                feature_names_accu=fnames_accu)
    else:
        if brokers:
            df_features = mjolnir.features.collect_kafka(
                df_hits,
                brokers=brokers,
                indices={wiki: '%s_content' % (wiki) for wiki in wikis},
                feature_definitions=mjolnir.features.enwiki_features(),
                feature_names_accu=fnames_accu)
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
                feature_definitions=mjolnir.features.enwiki_features(),
                feature_names_accu=fnames_accu)

    # collect the accumulator
    df_features.cache().count()

    if len(set(fnames_accu.value.values())) != 1:
        raise ValueError("Not all features were collected properly: " + str(fnames_accu.value))

    print 'Collected %d datapoints' % (fnames_accu.value.values()[0])
    # TODO: count and check that this value is sane, this would require computing the number
    # of request sent

    features = fnames_accu.value.keys()
    df_features = df_features.withColumn('features', mjolnir.spark.add_meta(df_features._sc, F.col('features'), {
            'features': features
        }))
    df_hits_with_features = (
        df_hits
        .join(df_features, how='inner', on=['wikiid', 'query', 'hit_page_id'])
        .withColumn('label', mjolnir.spark.add_meta(sc, F.col('label'), {
            'weightedNdcgAt10': weightedNdcgAt10,
            'ndcgAt10': ndcgAt10,
        })))

    df_hits_with_features.write.parquet(output_dir)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument(
        '-i', '--input', dest='input_dir', type=str,
        default='hdfs://analytics-hadoop/wmf/data/discovery/query_clicks/daily/year=*/month=*/day=*',
        help='Input path, prefixed with hdfs://, to query and click data')
    parser.add_argument(
        '-q', '--samples-per-wiki', dest='samples_per_wiki', type=int, default=1000000,
        help='The approximate number of rows in the final result per-wiki.')
    parser.add_argument(
        '-qe', '--sample-size-tolerance', dest='samples_size_tolerance', type=float, default=0.5,
        help='The tolerance between the --samples-per-wiki set and the actual number of rows fetched.')
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
        '-v', '--verbose', dest='verbose', default=False, action='store_true',
        help='Increase logging to INFO')
    parser.add_argument(
        '-vv', '--very-verbose', dest='very_verbose', default=False, action='store_true',
        help='Increase logging to DEBUG')
    parser.add_argument(
        '-f', '--feature-definitions', dest='ltr_feature_definitions', type=str, required=False,
        help='Name of the LTR plugin feature definitions (featureset:name[@store] or '
             + 'model:name[@store])')
    parser.add_argument(
        'wikis', metavar='wiki', type=str, nargs='+',
        help='A wiki to generate features and labels for')

    args = parser.parse_args(argv)
    if args.samples_size_tolerance < 0 or args.samples_size_tolerance > 1:
        raise ValueError('--sample-size-tolerance must be between 0 and 1')

    return dict(vars(args))


def main(argv=None):
    args = parse_arguments(argv)
    if args['very_verbose']:
        logging.basicConfig(level=logging.DEBUG)
    elif args['verbose']:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig()
    del args['verbose']
    del args['very_verbose']
    sc = SparkContext(appName="MLR: data collection pipeline")
    # spark info logging is incredibly spammy. Use warn to have some hope of
    # human decipherable output
    sc.setLogLevel('WARN')
    sqlContext = HiveContext(sc)
    run_pipeline(sc, sqlContext, **args)


if __name__ == "__main__":
    main()
