"""Helpful utilities for feature engineering"""
from __future__ import absolute_import
import numpy as np
import mjolnir.spark
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
        return Vectors.dense(np.append(raw, map(float, other)))
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


def explode_features(df):
    """Convert feature vector into individual columns

    Parameters
    ----------
    df : pyspark.sql.DataFrame

    Returns
    -------
    pyspark.sql.DataFrame
    """
    features = df.schema['features'].metadata['features']

    def extract_feature(features, idx):
        return float(features[idx])
    extract_feature_udf = F.udf(extract_feature, pyspark.sql.types.FloatType())
    cols = [extract_feature_udf('features', F.lit(idx)).alias(name) for idx, name in enumerate(features)]
    return df.select('*', *cols)
