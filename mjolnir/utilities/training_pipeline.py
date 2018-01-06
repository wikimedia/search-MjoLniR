"""
Example script demonstrating the training portion of the MLR pipeline.
This is mostly to demonstrate how everything ties together

To run:
    PYSPARK_PYTHON=venv/bin/python spark-submit \
        --jars /path/to/mjolnir-with-dependencies.jar \
        --artifacts 'mjolnir_venv.zip#venv' \
        path/to/training_pipeline.py
"""
from __future__ import absolute_import
import argparse
import datetime
import glob
import json
import logging
import mjolnir.feature_engineering
import mjolnir.training.xgboost
from mjolnir.utils import hdfs_open_read
import os
import pickle
from pyspark import SparkContext
from pyspark.sql import HiveContext
import sys


def run_pipeline(sc, sqlContext, input_dir, output_dir, wikis, initial_num_trees, final_num_trees, num_cv_jobs):
    with hdfs_open_read(os.path.join(input_dir, 'stats.json')) as f:
        stats = json.loads(f.read())

    wikis_available = set(stats['wikis'].keys())
    if wikis:
        missing = set(wikis).difference(wikis_available)
        if missing:
            raise Exception("Wikis not available: " + ", ".join(missing))
        wikis = wikis_available.intersection(wikis)
    else:
        wikis = stats['wikis'].keys()
    if not wikis:
        raise Exception("No wikis provided")

    for wiki in wikis:
        config = stats['wikis'][wiki]

        print 'Training wiki: %s' % (wiki)
        num_folds = config['num_folds']
        if num_cv_jobs is None:
            num_cv_jobs = num_folds

        # Add extension matching training type
        extension = ".xgb"

        # Add file extensions to all the folds
        folds = config['folds']
        for fold in folds:
            for partition in fold:
                for name, path in partition.items():
                    partition[name] = path + extension

        # "all" data with no splits
        all_paths = config['all']
        for partition in all_paths:
            for name, path in partition.items():
                partition[name] = path + extension

        tune_results = mjolnir.training.xgboost.tune(
            folds, config['stats'],
            num_cv_jobs=num_cv_jobs,
            train_matrix="train",
            initial_num_trees=initial_num_trees,
            final_num_trees=final_num_trees)

        print 'CV  test-ndcg@10: %.4f' % (tune_results['metrics']['cv-test'])
        print 'CV train-ndcg@10: %.4f' % (tune_results['metrics']['cv-train'])

        tune_results['metadata'] = {
            'wiki': wiki,
            'input_dir': input_dir,
            'training_datetime': datetime.datetime.now().isoformat(),
            'dataset': config['stats'],
        }

        # Train a model over all data with best params.
        best_params = tune_results['params'].copy()
        print 'Best parameters:'
        for param, value in best_params.items():
            print '\t%20s: %s' % (param, value)
        model = mjolnir.training.xgboost.train(
            all_paths, best_params, train_matrix="all")

        tune_results['metrics'] = {
            'train': model.summary().train()
        }
        print 'train-ndcg@10: %.5f' % (tune_results['metrics']['train'][-1])

        # Save the tune results somewhere for later analysis. Use pickle
        # to maintain the hyperopt.Trials objects as is. It might be nice
        # to write out a json version, but the Trials objects require
        # some more work before they can be json encoded.
        tune_output_pickle = os.path.join(output_dir, 'tune_%s.pickle' % (wiki))
        with open(tune_output_pickle, 'w') as f:
            # TODO: This includes special hyperopt and mjolnir objects, it would
            # be nice if those could be converted to something simple like dicts
            # and output json instead of pickle. This would greatly simplify
            # post-processing.
            f.write(pickle.dumps(tune_results))
            print 'Wrote tuning results to %s' % (tune_output_pickle)

        # Generate a feature map so xgboost can include feature names in the dump.
        # The final `q` indicates all features are quantitative values (floats).
        features = config['stats']['features']
        json_model_output = os.path.join(output_dir, 'model_%s.json' % (wiki))
        with open(json_model_output, 'wb') as f:
            f.write(model.dump(features))
            print 'Wrote xgboost json model to %s' % (json_model_output)
        # Write out the xgboost binary format as well, so it can be re-loaded
        # and evaluated
        model_output = os.path.join(output_dir, 'model_%s.xgb' % (wiki))
        model.saveModelAsLocalFile(model_output)
        print 'Wrote xgboost binary model to %s' % (model_output)
        print ''


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Train XGBoost ranking models')
    parser.add_argument(
        '-i', '--input', dest='input_dir', type=str, required=True,
        help='Input path, prefixed with hdfs://, to dataframe with labels and features')
    parser.add_argument(
        '-o', '--output', dest='output_dir', type=str, required=True,
        help='Path, on local filesystem, to directory to store the results of '
             'model training to.')
    parser.add_argument(
        '-c', '--cv-jobs', dest='num_cv_jobs', default=None, type=int,
        help='Number of cross validation folds to perform in parallel. Defaults to number '
             + 'of folds, to run all in parallel. If this is a multiple of the number '
             + 'of folds multiple cross validations will run in parallel.')
    parser.add_argument(
        '--initial-trees', dest='initial_num_trees', default=100, type=int,
        help='Number of trees to perform hyperparamter tuning with.  (Default: 100)')
    parser.add_argument(
        '--final-trees', dest='final_num_trees', default=None, type=int,
        help='Number of trees in the final ensemble. If not provided the value from '
             + '--initial-trees will be used.  (Default: None)')
    parser.add_argument(
        '-v', '--verbose', dest='verbose', default=False, action='store_true',
        help='Increase logging to INFO')
    parser.add_argument(
        '-vv', '--very-verbose', dest='very_verbose', default=False, action='store_true',
        help='Increase logging to DEBUG')
    parser.add_argument(
        'wikis', metavar='wiki', type=str, nargs='*',
        help='A wiki to perform model training for.')

    args = parser.parse_args(argv)
    return dict(vars(args))


def main(argv=None):
    args = parse_arguments(argv)
    if args['very_verbose']:
        logging.basicConfig(level=logging.DEBUG)
    elif args['verbose']:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig()
    del args['verbose']
    del args['very_verbose']
    # TODO: Set spark configuration? Some can't actually be set here though, so best might be to set all of it
    # on the command line for consistency.
    app_name = "MLR: training pipeline xgboost"
    if args['wikis']:
        app_name += ': ' + ', '.join(args['wikis'])
    sc = SparkContext(appName=app_name)
    sc.setLogLevel('WARN')
    sqlContext = HiveContext(sc)

    output_dir = args['output_dir']
    if os.path.exists(output_dir):
        logging.error('Output directory (%s) already exists' % (output_dir))
        sys.exit(1)

    # Maybe this is a bit early to create the path ... but should be fine.
    # The annoyance might be that an error in training requires deleting
    # this directory to try again.
    os.mkdir(output_dir)

    try:
        run_pipeline(sc, sqlContext, **args)
    except:  # noqa: E722
        # If the directory we created is still empty delete it
        # so it doesn't need to be manually re-created
        if not len(glob.glob(os.path.join(output_dir, '*'))):
            os.rmdir(output_dir)
        raise


if __name__ == "__main__":
    main()
