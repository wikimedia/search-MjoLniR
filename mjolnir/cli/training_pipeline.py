"""
Example script demonstrating the training portion of the MLR pipeline.
This is mostly to demonstrate how everything ties together

To run:
    PYSPARK_PYTHON=venv/bin/python spark-submit \
        --jars /path/to/mjolnir-with-dependencies.jar
        --artifacts 'mjolnir_venv.zip#venv' \
        path/to/training_pipeline.py
"""

import hyperopt
import mjolnir.dbn
import mjolnir.sampling
import mjolnir.features
import mjolnir.training.tuning
import mjolnir.training.xgboost
from pyspark import SparkContext
from pyspark.sql import HiveContext


def main(sc, sqlContext):
    # TODO: cli argument
    in_path = 'hdfs://analytics-hadoop/user/ebernhardson/mjolnir/hits_with_features'
    df_hits_with_features = sqlContext.read.parquet(in_path)

    # Explore a hyperparameter space
    best_params, trials = mjolnir.training.tuning.hyperopt(
        df_hits_with_features,
        mjolnir.training.xgboost.train,
        {
            # Generate 5 folds and run them all in parallel
            'num_folds': 5,
            'num_cv_jobs': 5,
            # Use lambdarank and eval with ndcg@10
            'objective': 'rank:ndcg',
            'eval_metric': 'ndcg@10',
            # gbtree options
            'eta': 0.3,
            'num_rounds': 100,
            'max_depth': hyperopt.hp.quniform('max_depth', 3, 10, 2),
            'min_child_weight': hyperopt.hp.quniform('min_child_weight', 100, 2000, 100),
        })

    # Train a model over all data with best params
    df_grouped, j_groups = mjolnir.training.xgboost.prep_training(
        df_hits_with_features, 10)
    best_params['groupData'] = j_groups
    # TODO: Maybe these don't belong in the params, and should instead have been kwargs?
    del best_params['num_folds']
    del best_params['num_cv_jobs']
    model = mjolnir.training.xgboost.train(df_grouped, best_params)

    print 'train-ndcg@10: %.3f' % (model.eval(df_grouped, j_groups))

    # Generate a featuremap so xgboost can include feature names in the dump.
    # The final `q` indicates all features are quantitative values (floats).
    features = df_hits_with_features.schema['features'].metadata['features']
    feat_map = ["%d %s q" % (i, fname) for i, fname in enumerate(features)]
    # TODO: this should be CLI argument as well
    with open('/home/ebernhardson/xgboost_model.json', 'wb') as f:
        f.write(model.dump("\n".join(feat_map)))


if __name__ == "__main__":
    sc = SparkContext(appName="MLR: training pipeline")
    sqlContext = HiveContext(sc)
    main(sc, sqlContext)
