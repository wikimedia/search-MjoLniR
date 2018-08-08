"""Helpful utilities for feature engineering"""
from __future__ import absolute_import
import numpy as np
import mjolnir.spark
from pyspark import SparkContext
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import functions as F
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


def quantiles(df, input_col):
    try:
        qds = QuantileDiscretizer(
            # 254 is used so the 255th can be inf
            numBuckets=254, inputCol=input_col, outputCol='bucketed',
            relativeError=1./2550, handleInvalid='error')
        return qds.fit(df).getSplits()
    except Exception as e:
        print(e)
        raise


def select_features(sc, df, all_features, n_features, pool, n_partitions=None, algo='mrmr'):
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
    pool : multiprocessing.dummy.Pool
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

    # InfoThSelector expects a double
    df = df.select(F.col('label').cast('double').alias('label'), 'features', *all_features)
    # TODO: QuantileDiscretizer in spark 2.2 can do multiple columns
    # at once instead of mapping over the list of features.
    df_quant = df.coalesce(10)
    quants = pool.map(lambda f: quantiles(df_quant, f), all_features)
    max_bin = max(len(x) for x in quants)
    # Build up an array that indicates the split positions
    # for the discretizer
    j_quants = SparkContext._gateway.new_array(
        sc._jvm.float, len(quants), max_bin)
    for i, q in enumerate(quants):
        for j, value in enumerate(q):
            j_quants[i][j] = value
        for j in range(max_bin - len(q)):
            j_quants[i][j] = float('inf')

    # Use this discretizer instead of bucketizer returned by
    # QuantileDiscretizer because it does all features together as a vector
    # which we need for InfoThSelector. Could potentially replace with
    # QuantileDiscretizer multi-column and vector assembler in spark 2.2
    discretizer = (
        sc._jvm.org.apache.spark.ml.feature.DiscretizerModel("???", j_quants)
        .setInputCol("features")
        .setOutputCol("features"))
    j_df_discretized = discretizer.transform(df._jdf)

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
    selector_model = selector.fit(j_df_discretized)
    selected_features = list(selector_model.selectedFeatures())
    return [all_features[i] for i in selected_features]
