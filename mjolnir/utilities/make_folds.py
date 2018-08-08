"""
Using the outputs of data_pipeline.py split our training data
into multiple folds with a test/train split for each. Save these
splits, to hdfs if requested, in the appropriate training algorithm
binary format. Additionally generates some statistics on the dataset.
"""
from __future__ import absolute_import
import argparse
import collections
import json
import logging
import multiprocessing.dummy
import math
import mjolnir.feature_engineering
import mjolnir.training.tuning
from mjolnir.utils import as_local_paths, as_output_file, as_output_files, \
                          hdfs_mkdir, hdfs_unlink
import os
import sys
from pyspark import SparkContext
from pyspark.sql import functions as F, HiveContext
import xgboost


def summarize_training_df(df, features, data_size):
    if data_size > 10000000:
        df = df.repartition(200)
    summary = collections.defaultdict(dict)
    for row in mjolnir.feature_engineering.explode_features(df, features).describe().collect():
        statistic = row.summary
        for field, value in (x for x in row.asDict().items() if x[0] != 'summary'):
            summary[field][statistic] = value
    return dict(summary)


def make_df_wiki_stats(df, features_metadata, data_size):
    metadata = dict(features_metadata)
    metadata.update(df.schema['label'].metadata)
    metadata.update({
        'num_observations': data_size,
        'num_queries': df.select('query').drop_duplicates().count(),
        'num_norm_queries': df.select('norm_query_id').drop_duplicates().count(),
        'summary': summarize_training_df(df, features_metadata['features'], data_size),
    })
    return metadata


def write_xgb(in_path, out_path):
    # Because xgboost hates us, reading a group file from disk
    # is only supported in cli. From the c_api we have to provide it.
    with open(in_path + '.query') as query:
        query_boundaries = [int(line) for line in query]
    # We cannot access the jvm from executors, only the driver,
    # which means jvm xgboost is not available. For this one limited
    # use case we must also have the python xgboost package.
    dmat = xgboost.DMatrix(in_path)
    dmat.set_group(query_boundaries)
    dmat.save_binary(out_path)


def write_wiki_folds(sc, df, num_workers, fold_col, path_format, features):
    # Trying to track down a problem with Rabit killing tests
    # in CI.
    if "pytest" in sys.modules:
        assert num_workers == 1

    def write_binaries(rows):
        # row is dict from split name (train/test) to path data can be found
        for pair in rows:
            i, row = pair
            with as_local_paths(row.values(), with_query=True) as local_inputs, \
                    as_output_files([path + '.xgb' for path in row.values()], 'wb') as local_outputs:
                for local_input, local_output in zip(local_inputs, local_outputs):
                    write_xgb(local_input, local_output.name)

    # Write out as text files from scala, much faster than shuffling to python
    writer = sc._jvm.org.wikimedia.search.mjolnir.DataWriter(sc._jsc, False)
    j_paths = writer.write(df._jdf, num_workers, path_format, fold_col)

    # Convert everything to python objects
    # in scala this is Array[Array[Map[String, String]]]
    all_paths = []
    for j_fold in j_paths:
        fold = []
        all_paths.append(fold)
        for j_partition in j_fold:
            partition = {str(k): str(v) for k, v in dict(j_partition).items()}
            fold.append(partition)

    # Enumerate gives a partition id used by partitionBy below
    # This isn't a partition id of the data, but the stage we are making.
    all_splits = list(enumerate(partition for fold in all_paths for partition in fold))
    # For all the emitted folds create binary data files
    sc.parallelize(all_splits, 1).partitionBy(len(all_splits), lambda x: x).foreachPartition(write_binaries)
    # Cleanup the text/query output, keeping only the binary data files
    hdfs_paths = []
    for i, partition in all_splits:
        for path in partition.values():
            for extension in ['', '.query']:
                hdfs_paths.append(path + extension)
    hdfs_unlink(*hdfs_paths)
    return all_paths


def write_wiki_all(*args):
    return write_wiki_folds(*args)[0]


def make_folds(sc, sqlContext, input_dir, output_dir, wikis, zero_features, num_folds, num_workers, max_executors):
    hdfs_mkdir(output_dir)
    df = sqlContext.read.parquet(input_dir) \
        .select('wikiid', 'query', 'features', 'label', 'norm_query_id')
    if wikis:
        df = df.where(F.col('wikiid').isin(wikis))

    counts = df.groupBy('wikiid').agg(F.count(F.lit(1)).alias('n_obs')).collect()
    counts = {row.wikiid: row.n_obs for row in counts}

    if not wikis:
        wikis = counts.keys()
    else:
        missing = set(wikis).difference(counts.keys())
        for wiki in missing:
            print('No observations available for ' + wiki)
        wikis = list(set(wikis).intersection(counts.keys()))
    if not wikis:
        raise Exception('No wikis provided')

    # sort to descending size, so mapping over them does the largest first
    wikis.sort(reverse=True, key=lambda wiki: counts[wiki])

    if zero_features:
        df = mjolnir.feature_engineering.zero_features(df, zero_features)
    if max_executors is None:
        max_executors = num_workers

    pool_size = int(math.floor(max_executors / float(num_workers)))
    pool = multiprocessing.dummy.Pool(pool_size)

    df_fold = (
        mjolnir.training.tuning.group_k_fold(df, num_folds)
        .repartition(200, 'wikiid', 'query')
        .sortWithinPartitions('wikiid', 'query', F.col('label').asc()))

    try:
        df_fold.cache()
        df_fold.count()

        wiki_stats = {}
        for wiki in wikis:
            df_wiki = df_fold.where(F.col('wikiid') == wiki).drop('wikiid')
            path_format = os.path.join(output_dir, wiki + '.%s.f%s.p%d')
            metadata = dict(df.schema['features'].metadata)
            if 'wiki_features' in metadata:
                metadata['features'] = metadata['wiki_features'][wiki]
                del metadata['wiki_features']
            wiki_stats[wiki] = {
                'all': pool.apply_async(
                    write_wiki_all,
                    (sc, df_wiki, num_workers, None, path_format, metadata['features'])),
                'folds': pool.apply_async(
                    write_wiki_folds,
                    (sc, df_wiki, num_workers, 'fold', path_format, metadata['features'])),
                'stats': pool.apply_async(make_df_wiki_stats, (df_wiki, metadata, counts[wiki])),
            }

        wiki_stats = {wiki: {k: v.get() for k, v in stats.items()} for wiki, stats in wiki_stats.items()}
        for wiki in wikis:
            wiki_stats[wiki]['num_folds'] = num_folds
            wiki_stats[wiki]['num_workers'] = num_workers
    finally:
        df_fold.unpersist()

    with as_output_file(os.path.join(output_dir, 'stats.json'), 'w') as f:
        f.write(json.dumps({
            'input_dir': input_dir,
            'wikis': wiki_stats,
        }))


def arg_parser():
    parser = argparse.ArgumentParser(description='Prepare XGB binary matrices')
    parser.add_argument(
        '-i', '--input', dest='input_dir', type=str, required=True,
        help='Input path, prefixed with hdfs://, to dataframe with labels and features')
    parser.add_argument(
        '-o', '--output-dir', dest='output_dir', type=str, required=True,
        help='Output path, prefixed with hdfs://, to store binary matrices')
    parser.add_argument(
        '-f', '--num-folds', dest='num_folds', type=int, default=5,
        help='The number of folds to split the data into')
    parser.add_argument(
        '-x', '--max-executors', dest='max_executors', type=int, default=None,
        help='The maximum number of executors to use to write out folds')
    parser.add_argument(
        '-w', '--num-workers', dest='num_workers', type=int, default=1,
        help='The number of workers used to train a single model')
    parser.add_argument(
        '-z', '--zero-features', dest='zero_features', type=str, required=False, default=None,
        help='TODO')
    parser.add_argument(
        'wikis', metavar='wiki', type=str, nargs='*',
        help='List of wikis to build matrices for')
    return parser


def main(**kwargs):
    app_name = 'MLR: writer binary folded datasets'
    if kwargs['wikis']:
        app_name += ': ' + ', '.join(kwargs['wikis'])
    sc = SparkContext(appName=app_name)
    sc.setLogLevel('WARN')
    sqlContext = HiveContext(sc)

    make_folds(sc, sqlContext, **kwargs)


if __name__ == '__main__':
    logging.basicConfig()
    kwargs = dict(vars(arg_parser().parse_args()))
    main(**kwargs)
