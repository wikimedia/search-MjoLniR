"""Helpful utilities for feature engineering"""
from __future__ import absolute_import
import numpy as np
import mjolnir.spark
from pyspark import SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.wrapper import JavaParams
from pyspark.sql import DataFrame, functions as F
import pyspark.sql.types


def append_features(df, *cols):
    """Append features from columns to the features vector.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    cols : list of str

    Returns
    -------
    pyspark.sql.DataFrame
    """
    def add_features(feat, *other):
        raw = feat.toArray()
        return Vectors.dense(np.append(raw, list(map(float, other))))
    add_features_udf = F.udf(add_features, VectorUDT())
    new_feat_list = df.schema['features'].metadata['features'] + cols
    return df.withColumn('features', mjolnir.spark.add_meta(
        df._sc, add_features_udf('features', *cols), {'features': new_feat_list}))


def zero_features(df, *feature_names):
    """Zero out features in the feature vector.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    feature_names : list of str

    Returns
    -------
    pyspark.sql.DataFrame
    """
    features = df.schema['features'].metadata['features']
    idxs = [features.index(name) for name in feature_names]

    def zero_features(feat):
        raw = feat.toArray()
        for idx in idxs:
            raw[idx] = 0.
        return Vectors.dense(raw)
    zero_features_udf = F.udf(zero_features, VectorUDT())
    return df.withColumn('features', mjolnir.spark.add_meta(
        df._sc, zero_features_udf('features'), {'features': features}))


def explode_features(df, features=None):
    """Convert feature vector into individual columns

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    features : list of str or None

    Returns
    -------
    pyspark.sql.DataFrame
    """
    if features is None:
        features = df.schema['features'].metadata['features']

    def extract_feature(features, idx):
        return float(features[idx])
    extract_feature_udf = F.udf(extract_feature, pyspark.sql.types.FloatType())
    cols = [extract_feature_udf('features', F.lit(idx)).alias(name) for idx, name in enumerate(features)]
    return df.select('*', *cols)


def _bucketize(df, input_cols):
    def j_str_arr(arr):
        gateway = SparkContext._gateway
        j_str = gateway.jvm.java.lang.String
        j_arr = gateway.new_array(j_str, len(arr))
        for i, val in enumerate(arr):
            j_arr[i] = val
        return j_arr

    output_cols = ['{}-bucketed'.format(x) for x in input_cols]
    # Sadly the multi-col versions are only in scala, pyspark doesn't
    # have them yet.
    j_bucketizer = (
        JavaParams._new_java_obj("org.apache.spark.ml.feature.QuantileDiscretizer")
        .setInputCols(j_str_arr(input_cols))
        .setOutputCols(j_str_arr(output_cols))
        .setNumBuckets(254)
        .setRelativeError(1/2550)
        .setHandleInvalid('error')
        .fit(df._jdf))
    j_df_bucketized = j_bucketizer.transform(df._jdf)
    df_bucketized = DataFrame(j_df_bucketized, df.sql_ctx).drop(*input_cols)
    # Now we need to assemble the bucketized values into vector
    # form for the feature selector to work with.
    assembler = VectorAssembler(
            inputCols=output_cols, outputCol='features')
    return assembler.transform(df_bucketized).drop(*output_cols)


def select_features(sc, df, all_features, n_features, n_partitions=None, algo='mrmr'):
    """Select a set of features from a dataframe

    Parameters
    ----------
    sc : pyspark.SparkContext
    df : pyspark.sql.DataFrame
        For reasonable performance this should be read from disk
        in parquet format so stages can pull a single column instead
        of the whole dataset.
    all_features : list of str
        List of columns in df to select from in the same order as
        stored in the 'features' column vector.
    n_features : int
        The number of features to select
    algo : str
        The algorithm to use in InfoThSelector

    Returns
    -------
    list of str
    """
    if n_features >= len(all_features):
        # Requested more features than we have, return them all
        return all_features

    if n_partitions is None:
        # The lib suggests we should have no more than 1 partition
        # per feature, and ideally less. 3 is arbitrary.
        features_per_partition = 3
        n_partitions = int(len(all_features)/features_per_partition)

    # InfoThSelector expects a double. We expect our input to have
    # each feature as a separate column.
    df = df.select(F.col('label').cast('double').alias('label'), *all_features)
    # Features must be separate columns for the bucketizing. Once bucketized
    # the features vector is reassembled.
    df_bucketized = _bucketize(df, all_features)

    selector = (
        sc._jvm.org.apache.spark.ml.feature.InfoThSelector()
        .setSelectCriterion(algo)
        # TODO: How should this be set? On our largest datasets this is ~40M
        # per partition and everything seems to work reasonably. On the smaller
        # ones this results in 500kB partitions and we probably waste a bunch
        # of orchestration time.
        .setNPartitions(n_partitions)
        .setNumTopFeatures(n_features)
        .setFeaturesCol("features"))
    selector_model = selector.fit(df_bucketized._jdf)
    selected_features = list(selector_model.selectedFeatures())
    return [all_features[i] for i in selected_features]
