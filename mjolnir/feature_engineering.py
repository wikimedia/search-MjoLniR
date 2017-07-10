import numpy as np
from pyspark.ml.linalg import Vectors, VectorUDT

def append_features(df, *cols):
    def add_features(feat, *other):
        raw = feat.toArray()
        return Vectors.dense(np.append(raw, map(float, other)))
    add_features_udf = F.udf(add_features, VectorUDT())
    new_feat_list = df.schema['features'].metadata['features'] + cols
    return df.withColumn('features', mjolnir.spark.add_meta(df._sc, add_features_udf('features', *cols),
                                                            {'features': new_feat_list}))

