"""CLI application definition helpers

Implements CLI argument handling for transformation
utilities.
"""
from argparse import ArgumentParser
from collections import OrderedDict
from functools import wraps
import json
import os
from typing import cast, Callable, Dict, List, Mapping, Optional
import uuid

try:
    from pyspark.sql import DataFrame, SparkSession
except ImportError:
    import findspark
    findspark.init()
    from pyspark.sql import DataFrame, SparkSession

from mjolnir.utils import as_output_file, hdfs_open_read, hdfs_rmdir
import mjolnir.transform as mt


METADATA_FILE_NAME = '_METADATA.JSON'


def write_metadata(dir_path, metadata) -> None:
    file_path = os.path.join(dir_path, METADATA_FILE_NAME)
    with as_output_file(file_path, overwrite=True) as f:
        json.dump(metadata, f)


def read_metadata(dir_path) -> Mapping:
    file_path = os.path.join(dir_path, METADATA_FILE_NAME)
    with hdfs_open_read(file_path) as f:
        return json.load(f)


def _wiki_features(df: DataFrame, wiki: str) -> List[str]:
    meta = df.schema['features'].metadata
    if 'wiki_features' in meta:
        return meta['wiki_features'][wiki]
    else:
        return meta['features']


def feature_vector_stats(df: DataFrame) -> Mapping:
    """Calculate stats of feature vector partitions necessary for executor auto-sizing

    Expects the input dataframe to either be cached, or read directly from
    disk. Will iterate over the dataframe multiple times. Signature matches
    stats_fn of write_partition function.
    """
    num_obs = {row['wikiid']: row['count']
               for row in df.groupBy('wikiid').count().collect()}
    features = {wiki: _wiki_features(df, wiki) for wiki in num_obs.keys()}

    return {'num_obs': num_obs, 'wiki_features': features}


class HivePartition:
    def __init__(self, spark, table, partition_spec, direct_parquet_read=False):
        self._spark = spark
        self.table = table
        self.partition_spec = partition_spec
        self._df = cast(Optional[DataFrame], None)
        self._metadata = cast(Optional[Mapping], None)
        self._direct_parquet_read = direct_parquet_read

    @property
    def df(self) -> DataFrame:
        """Pyspark dataframe for specified partition"""
        if self._df is None:
            self._df = mt.read_partition(
                self._spark, self.table, self.partition_spec,
                direct_parquet_read=self._direct_parquet_read)
        return self._df

    @property
    def input_dir(self) -> str:
        """Path to partition data"""
        paths = list(self.df._jdf.inputFiles())  # type: ignore
        dirs = {os.path.dirname(path) for path in paths}
        if len(dirs) != 1:
            raise Exception('multiple paths for [{}] [{}]: {}'.format(
                self.table, self.partition_spec, ','.join(dirs)))
        return next(iter(dirs))

    @property
    def metadata(self) -> Mapping:
        """Mjolnir specific metadata for specified partition"""
        if self._metadata is None:
            self._metadata = read_metadata(self.input_dir)
        return self._metadata

    def partition_value(self, key):
        """Lookup value in partition key/value pairs"""
        return dict(self.partition_spec)[key]


class CallQueue:
    def __init__(self):
        self._queue = []

    def __call__(self, *args, **kwargs):
        while self._queue:
            fn = self._queue.pop(0)
            fn(*args, **kwargs)

    def append(self, fn):
        self._queue.append(fn)
        return fn


class Cli:
    def __init__(self, name: str, transformer: Callable, parser: ArgumentParser):
        self.name = name
        self.transformer = transformer
        self.parser = parser
        self._post_process_args = CallQueue()
        self._post_process_transform = CallQueue()
        self._cleanup = CallQueue()
        # Default args for all scripts
        self.add_argument('--date', required=True)

    def __call__(self, **kwargs):
        spark = SparkSession.builder.getOrCreate()
        self._post_process_args(spark, kwargs)

        try:
            maybe_df = self.transformer(spark=spark, **kwargs)
            self._post_process_transform(maybe_df, kwargs)
        finally:
            self._cleanup()

    @wraps(ArgumentParser.add_argument)
    def add_argument(self, *args, **kwargs):
        return self.parser.add_argument(*args, **kwargs)

    def require_kafka_arguments(self):
        self.add_argument('--brokers', required=True)
        self.add_argument('--topic-request', required=True)
        self.add_argument('--topic-response', required=True)

    def require_daily_input_table(self):
        @self._post_process_args.append
        def post(spark: SparkSession, kwargs: Dict) -> None:
            kwargs['df_in'] = spark.read.table(kwargs['input_table']) \
                .drop('year', 'month', 'day')

        self.add_argument('--input-table', required=True)

    def require_query_clicks_partition(self):
        @self._post_process_args.append
        def post(spark: SparkSession, kwargs: Dict) -> None:
            kwargs['query_clicks'] = HivePartition(
                spark, kwargs['clicks_table'], {
                    'date': kwargs['date'],
                })

        self.add_argument('--clicks-table', required=True)

    def require_query_clustering_partition(self):
        @self._post_process_args.append
        def post(spark: SparkSession, kwargs: Dict) -> None:
            kwargs['query_clustering'] = HivePartition(
                spark, kwargs['clustering_table'], {
                    'date': kwargs['date'],
                    'algorithm': kwargs['clustering_algorithm'],
                })

        self.add_argument('--clustering-table', required=True)
        self.add_argument('--clustering-algorithm', required=True)

    def require_labeled_query_page_partition(self):
        @self._post_process_args.append
        def post(spark: SparkSession, kwargs: Dict) -> None:
            kwargs['labeled_query_page'] = HivePartition(
                spark, kwargs['labels_table'], {
                    'date': kwargs['date'],
                    'algorithm': kwargs['labeling_algorithm'],
                })

        self.add_argument('--labels-table', required=True)
        self.add_argument('--labeling-algorithm')

    def require_feature_vectors_partition(self):
        @self._post_process_args.append
        def post(spark: SparkSession, kwargs: Dict) -> None:
            kwargs['feature_vectors'] = HivePartition(
                spark, kwargs['feature_vectors_table'], {
                    'date': kwargs['date'],
                    'feature_set': kwargs['feature_set']
                }, direct_parquet_read=True)

        self.add_argument('--feature-vectors-table', required=True)
        self.add_argument('--feature-set', required=True)

    def require_training_files_partition(self):
        @self._post_process_args.append
        def post(spark: SparkSession, kwargs: Dict) -> None:
            kwargs['training_files'] = read_metadata(
                kwargs['training_files_path'])

        self.add_argument('--training-files-path', required=True)

    def require_model_parameters_partition(self, only_table=False):
        self.add_argument('--model-parameters-table', required=True)
        if not only_table:
            raise NotImplementedError("TODO")

    @staticmethod
    def _resolve_partition_spec(kwargs, partition_spec_spec) -> Dict[str, str]:
        # Bypass, typically repeating an input partitions
        # partition spec.
        if callable(partition_spec_spec):
            return OrderedDict(partition_spec_spec(kwargs))
        # Partition always starts with date
        partition_spec = cast(Dict[str, str], OrderedDict())
        partition_spec['date'] = kwargs['date']
        for k, v in partition_spec_spec:
            partition_spec[k] = v.format(**kwargs)
        return partition_spec

    def require_output_table(
        self, partition_spec_spec, metadata_fn=None,
        mode='overwrite',
    ):
        @self._post_process_transform.append
        def post(df: DataFrame, kwargs: Dict):
            mt.write_partition(
                df, kwargs['output_table'], kwargs['output_path'],
                self._resolve_partition_spec(kwargs, partition_spec_spec),
                mode=mode)
            if metadata_fn is not None:
                spark = df.sql_ctx.sparkSession
                metadata = metadata_fn(spark.read.parquet(kwargs['output_path']))
                write_metadata(kwargs['output_path'], metadata)

        self.add_argument('--output-table', required=True)
        self.add_argument('--output-path', required=True)

    def require_temp_dir(self):
        state = {}

        @self._post_process_args.append
        def post(spark: SparkSession, kwargs: Dict):
            state['temp_dir'] = '{}-temp-{}'.format(
                kwargs['output_path'], uuid.uuid1())
            kwargs['temp_dir'] = state['temp_dir']

        @self._cleanup.append
        def cleanup():
            try:
                hdfs_rmdir(state['temp_dir'])
            except Exception:
                pass

    def require_output_metadata(self):
        @self._post_process_transform.append
        def post(metadata: Mapping, kwargs: Dict):
            write_metadata(kwargs['output_path'], metadata)

        self.add_argument('--output-path', required=True)
