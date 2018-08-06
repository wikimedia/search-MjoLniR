"""
Transforms user behaviour logs into a labeled training dataset
"""

from __future__ import absolute_import
import argparse
import datetime
import json
import logging
import mjolnir.dbn
import mjolnir.metrics
import mjolnir.norm_query
import mjolnir.features
import mjolnir.sampling
import os
from pyspark import SparkContext
from pyspark.sql import HiveContext, functions as F
import requests


def collect_features(sc, sqlContext, input_dir, output_dir, wikis,
                     search_cluster, brokers, ltr_feature_definitions,
                     session_factory=requests.Session):

    df_hits = sqlContext.read.parquet(input_dir)

    # Collect features for all known query, hit_page_id combinations.
    df_features, fnames_accu = mjolnir.features.collect(
        df_hits,
        url_list=mjolnir.cirrus.SEARCH_CLUSTERS[search_cluster] if search_cluster else None,
        model=ltr_feature_definitions,
        brokers=brokers,
        indices={wiki: '%s_content' % (wiki) for wiki in wikis},
        session_factory=session_factory)

    # collect the accumulator
    df_features.cache().count()

    num_rows_collected = set(fnames_accu.value.values())
    if len(num_rows_collected) != 1:
        raise ValueError("Not all features were collected properly: " + str(fnames_accu.value))
    num_rows_collected = num_rows_collected.pop()
    print('Collected %d datapoints' % (num_rows_collected))
    # TODO: count and check that this value is sane, this would require computing the number
    # of request sent

    features = list(fnames_accu.value.keys())
    df_hits_with_features = (
        df_hits
        # TODO: We almost don't need the join and could defer it to something
        # that does, basically make_folds. But it's convenient to think of each
        # utility as adding to the previous and having a simple line as the
        # dependency chain.
        .join(df_features, how='inner', on=['wikiid', 'query', 'hit_page_id'])
        .withColumn('features', mjolnir.spark.add_meta(sc, F.col('features'), {
            'features': features,
            'feature_definitions': ltr_feature_definitions,
            'collected_at': datetime.datetime.now().isoformat(),
            'used_kafka': brokers is not None,
            'search_cluster': search_cluster if search_cluster else "",
            # TODO: Where does this metadata go? It seems a bit more top-level
            # but could be useful to remember.
            'input_dir': input_dir,
            'output_dir': output_dir,
        })))

    df_hits_with_features.write.parquet(output_dir)

    # Emit some statistics that will allow the spark utility to automatically
    # size the executors for building datasets out of this.
    counts = (
        sqlContext.read.parquet(output_dir)
        .groupBy('wikiid')
        .agg(F.count(F.lit(1)).alias('num_obs'))
        .collect())
    stats_path = os.path.join(output_dir, '_stats.json')
    with mjolnir.utils.as_output_file(stats_path) as f:
        f.write(json.dumps({
            'num_features': len(features),
            'num_obs': {row.wikiid: row.num_obs for row in counts}
        }))


def arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-i', '--input', dest='input_dir', type=str, required=True,
        help='Input path, prefixed with hdfs://, to query and click data')
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
        '-f', '--feature-definitions', dest='ltr_feature_definitions', type=str, required=True,
        help='Name of the LTR plugin feature definitions (featureset:name[@store] or '
             + 'model:name[@store])')
    parser.add_argument(
        'wikis', metavar='wiki', type=str, nargs='+',
        help='A wiki to collect features for')
    return parser


def main(**kwargs):
    if len(kwargs['brokers']) == 1 and ',' in kwargs['brokers'][0]:
        kwargs['brokers'] = kwargs['brokers'][0].split(',')

    sc = SparkContext(appName="MLR: data collection pipeline")
    # spark info logging is incredibly spammy. Use warn to have some hope of
    # human decipherable output
    sc.setLogLevel('WARN')
    sqlContext = HiveContext(sc)
    collect_features(sc, sqlContext, **kwargs)


if __name__ == "__main__":
    logging.basicConfig()
    kwargs = dict(vars(arg_parser().parse_args()))
    main(**kwargs)
