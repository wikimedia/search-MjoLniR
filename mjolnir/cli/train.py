from argparse import ArgumentParser
import os
from typing import Callable, List, Mapping

import numpy as np

from mjolnir.cli.helpers import Cli, HivePartition
import mjolnir.training.xgboost
import mjolnir.transform as mt
from mjolnir.utils import as_output_file

from pyspark.sql import functions as F, Row, SparkSession


def decide_best_params(all_params: List[Row]) -> Row:
    # Input rows match mt.ModelParameters
    best_loss_idx = np.argmin(row['loss'] for row in all_params)
    initial_best_params = all_params[best_loss_idx]
    # TODO: Something better than straight argmin, likely there are several
    # quite similar sets of params that could be chosen between.
    return initial_best_params


def train(
    spark: SparkSession,
    date: str,
    output_path: str,

    training_files: Mapping,
    model_parameters_table: str,

    remote_feature_set: str,
    **kwargs
) -> Mapping:
    wiki = training_files['metadata']['wikiid']
    features = training_files['metadata']['features']

    folds = [row for row in training_files['rows']
             if row['vec_format'] == 'xgboost' and row['split_name'] == 'all']
    if len(folds) != 1:
        raise Exception('Expected single "all" training file in xgb format')
    data_path = folds[0]['path']

    model_parameters = HivePartition(
        spark, model_parameters_table, training_files['partition_spec'])

    df_model_params = (
        model_parameters.df
        # wikiid is part of the partition spec, so it was dropped. But it's
        # also part of all the schemas. This is currently an isolated problem,
        # so simply add it back in.
        .withColumn('wikiid', F.lit(training_files['partition_spec']['wikiid']))
    )

    mt.check_schema(df_model_params, mt.ModelParameters)
    transformer = mt.restrict_wikis([wiki])
    all_model_param_evals = transformer(df_model_params).collect()
    # best_params_eval is an mt.ModelParameters row
    best_param_eval = decide_best_params(all_model_param_evals)

    model = mjolnir.training.xgboost.train(
        {'all': data_path}, best_param_eval.params, 'all', spark=spark)

    # Write model to output directory in binary format, readable by xgboost
    # TODO: Where do feature names go?
    model.saveBoosterAsHadoopFile(os.path.join(output_path, 'model.xgb'))
    # Write model to output directory in json format, readable by elasticsearch
    with as_output_file(os.path.join(output_path, 'model.json')) as f:
        f.write(model.dump(features=features, with_stats=True))

    wikiid = training_files['partition_spec']['wikiid']

    return {
        'partition_spec': training_files['partition_spec'],
        'metadata': dict(training_files['metadata'], **{
            'training_file_path': data_path,
            'model_params': best_param_eval.params,
        }),
        # Parameters expected by model upload.
        'upload': {
            'wikiid': wikiid,
            # Is this acceptable? I dunno...
            'model_name': '{}-{}-{}'.format(wikiid, date, remote_feature_set),
            'model_type': 'model/xgboost+json',
            'feature_definition': 'featureset:{}'.format(remote_feature_set),
            'features': features,
            # TODO: Un-hardcode parameters used by features
            'validation_params': {
                'query_string': 'example query',
            }
        }
    }


def configure(parser: ArgumentParser) -> Callable:
    main = Cli('train', train, parser)
    # I/O
    main.require_training_files_partition()
    # We source the partition spec from training files metadata,
    # it didn't really fit as a generic solution so ask for
    # only the table name.
    main.require_model_parameters_partition(only_table=True)
    main.require_output_metadata()
    # Nothing to do with training, but we need it in the output
    # metadata to feed into the upload stage.
    main.add_argument('--remote-feature-set', required=True)
    return main
