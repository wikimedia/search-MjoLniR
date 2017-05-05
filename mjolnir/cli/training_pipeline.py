"""
Example script demonstrating the training portion of the MLR pipeline.
This is mostly to demonstrate how everything ties together

To run:
    PYSPARK_PYTHON=venv/bin/python spark-submit \
        --jars /path/to/mjolnir-with-dependencies.jar
        --artifacts 'mjolnir_venv.zip#venv' \
        path/to/training_pipeline.py
"""

import mjolnir.dbn
import mjolnir.sampling
import mjolnir.features
import mjolnir.training.tuning
import mjolnir.training.xgboost
import pickle
from pyspark import SparkContext
from pyspark.sql import HiveContext


def main(sc, sqlContext):
    # TODO: cli argument
    in_path = 'hdfs://analytics-hadoop/user/ebernhardson/mjolnir/hits_with_features'
    df_hits_with_features = sqlContext.read.parquet(in_path)

    # Explore a hyperparameter space. Skip the most expensive part of tuning,
    # increasing the # of trees, with target_node_evaluations=None
    tune_results = mjolnir.training.xgboost.tune(
        df_hits_with_features, target_node_evaluations=None)

    # Save the tune results somewhere for later analysis. Use pickle
    # to maintain the hyperopt.Trials objects as is.
    # TODO: Path should be CLI argument
    with open('/home/ebernhardson/xgboost_training.pickle', 'w') as f:
        f.write(pickle.dumps(tune_results))

    # Train a model over all data with best params
    best_params = tune_results['params']
    df_grouped, j_groups = mjolnir.training.xgboost.prep_training(
        df_hits_with_features, 10)
    best_params['groupData'] = j_groups
    model = mjolnir.training.xgboost.train(df_grouped, best_params)

    print 'train-ndcg@10: %.3f' % (model.eval(df_grouped, j_groups))

    # Generate a feature map so xgboost can include feature names in the dump.
    # The final `q` indicates all features are quantitative values (floats).
    features = df_hits_with_features.schema['features'].metadata['features']
    feat_map = ["%d %s q" % (i, fname) for i, fname in enumerate(features)]
    # TODO: this path should be CLI argument as well
    with open('/home/ebernhardson/xgboost_model.json', 'wb') as f:
        f.write(model.dump("\n".join(feat_map)))


if __name__ == "__main__":
    # TODO: Set spark configuration? Some can't actually be set here though, so best might be to set all of it
    # on the command line for consistency.
    sc = SparkContext(appName="MLR: training pipeline")
    sqlContext = HiveContext(sc)
    main(sc, sqlContext)
