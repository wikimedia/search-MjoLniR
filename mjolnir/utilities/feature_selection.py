"""
Reduce the number of features used in a dataset
"""

from __future__ import absolute_import
import argparse
from functools import reduce
import logging
import mjolnir.feature_engineering
import mjolnir.spark
import mjolnir.utils
import multiprocessing.dummy
from pyspark import SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import HiveContext
from pyspark.sql import functions as F


def run_pipeline(sc, sqlContext, input_dir, output_dir, algo, num_features, pre_selected, wikis):
    """Reduce the number of features used in a dataset

    Applies feature selection from sramirez:spark-infotheoretic-feature-selection
    to the collected dataset to reduce the number of features models will be
    trained with. Fewer features means less time/memory needed for training, and
    less time needed at ranking time to collect feature vectors.
    """
    df_in = sqlContext.read.parquet(input_dir)
    if wikis:
        df_in = df_in.where(df_in['wikiid'].isin(wikis))
    else:
        wikis = [row.wikiid for row in df_in.select('wikiid').drop_duplicates().collect()]

    all_features = df_in.schema['features'].metadata['features']

    exploded_output_dir = output_dir + '-temp'
    try:
        # Parquet allows reading a single column at a time which is perfect
        # for our process to quantize individual features
        (mjolnir.feature_engineering.explode_features(df_in)
            .write.partitionBy('wikiid').parquet(exploded_output_dir))

        df_exploded = sqlContext.read.parquet(exploded_output_dir)

        default_cols = [x for x in df_in.columns if x != 'features']
        wiki_features = {}
        quantile_pool = multiprocessing.dummy.Pool(80)

        def process_wiki(wiki):
            df_wiki = df_exploded.where(F.col('wikiid') == wiki)
            if pre_selected:
                selected = pre_selected
            else:
                selected = mjolnir.feature_engineering.select_features(
                    sc, df_wiki, all_features, num_features, quantile_pool,
                    algo=algo)
            df_selected = df_wiki.select(*(default_cols + selected))
            assembler = VectorAssembler(
                inputCols=selected, outputCol='features')
            wiki_features[wiki] = selected
            return assembler.transform(df_selected).drop(*selected)

        # Attempt to improve utilization by running a couple at
        # a time. utilization is still poor during feature selection.
        pool = multiprocessing.dummy.Pool(2)
        assembled = pool.imap_unordered(process_wiki, wikis)

        (reduce(lambda a, b: a.union(b), assembled)
            .coalesce(400)
            .withColumn('features', mjolnir.spark.add_meta(sc, F.col('features'), dict(
                df_in.schema['features'].metadata, **{
                    'wiki_features': wiki_features
                })))
            .write.parquet(output_dir))
    finally:
        try:
            mjolnir.utils.hdfs_rmdir(exploded_output_dir)
        except Exception:
            pass


def arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-i', '--input', dest='input_dir', type=str,
        default='hdfs://analytics-hadoop/wmf/data/discovery/query_clicks/daily/year=*/month=*/day=*',
        help='Input path, prefixed with hdfs://, to query and click data')
    # TODO: Per-wiki feature selection algo?
    parser.add_argument(
        '-a', '--algo', dest='algo', type=str, default='mrmr', choices=['mrmr'],
        help='Feature selection algorithm to apply')
    parser.add_argument(
        '-n', '--num-features', dest='num_features', type=int, default=50,
        help='The number of features to keep per-wiki')
    parser.add_argument(
        '-o', '--output-dir', dest='output_dir', type=str, required=True,
        help='Output path, prefixed with hdfs://, to write resulting dataframe to')
    parser.add_argument(
        '-f', '--features', dest='pre_selected', type=lambda x: x.split(','), required=False, default=None,
        help='Comma separated list of pre-selected features. Optional')
    parser.add_argument(
        'wikis', metavar='wiki', type=str, nargs='+',
        help='A wiki to generate features and labels for')
    return parser


def main(**kwargs):
    sc = SparkContext(appName="MLR: feature selection")
    # spark info logging is incredibly spammy. Use warn to have some hope of
    # human decipherable output
    sc.setLogLevel('WARN')
    sqlContext = HiveContext(sc)
    run_pipeline(sc, sqlContext, **kwargs)


if __name__ == "__main__":
    logging.basicConfig()
    kwargs = dict(vars(arg_parser().parse_args()))
    main(**kwargs)
