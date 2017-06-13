"""
Example script demonstrating the training portion of the MLR pipeline.
This is mostly to demonstrate how everything ties together

To run:
    PYSPARK_PYTHON=venv/bin/python spark-submit \
        --jars /path/to/mjolnir-with-dependencies.jar
        --artifacts 'mjolnir_venv.zip#venv' \
        path/to/training_pipeline.py
"""

import mjolnir.training
import mjolnir.training.xgboost
from pyspark import SparkContext
from pyspark.sql import HiveContext


def main(sc, sqlContext):
    # TODO: cli argument
    in_path = 'hdfs://analytics-hadoop/user/ebernhardson/mjolnir/hits_with_features'
    df_hits_with_features = sqlContext.read.parquet(in_path)

    # Doesn't have to be done ahead of time, but if the data is used multiple times
    # (eg train and eval) then it should to save work.
    df_grouped, j_groups = mjolnir.training.xgboost.prep_training(
        df_hits_with_features, 10)

    # Train a model
    # Note that this might be best done in a separate spark context, with different options.
    # All of the above code will use 1 cpu per task, but xgboost can use multiple cores per task.
    # To take advantage of multiple cores you should use spark.task.cpus=n in the spark config.
    model = mjolnir.training.xgboost.train(df_grouped, {
        'num_rounds': 100,
        'num_workers': 10,
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg@10',
        'eta': 0.3,
        'max_depth': 6,
        'groupData': j_groups,
    })

    print 'train-ndcg@10: %.3f' % (model.eval(df_grouped, j_groups))

    # dumped is now a list of strings each containing json of one
    # tree in the ensemble.
    dumped = model.dump()
    with open('/home/ebernahrdson/xgboost_model.json', 'wb') as f:
        f.writelines(dumped)


if __name__ == "__main__":
    sc = SparkContext(appName="MLR: training pipeline")
    sqlContext = HiveContext(sc)
    main(sc, sqlContext)
