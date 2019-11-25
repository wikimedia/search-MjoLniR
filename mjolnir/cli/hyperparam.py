from argparse import ArgumentParser
import itertools
from typing import cast, Callable, Dict, List, Mapping, Optional, Sequence
import uuid

from mjolnir.cli.helpers import Cli
import mjolnir.training.xgboost
import mjolnir.transform as mt

from pyspark.sql import DataFrame, SparkSession


def regroup_folds_metadata(raw_data: List[Mapping]) -> List[Mapping[str, str]]:
    """Reshape rows into nested format expected by training

    Input rows match the mt.TrainingFiles schema. Output matches expectation
    of mjolnir.training.xgboost.tune
    """
    def groupby(data, key_value):
        def key(row):
            return row[key_value]
        return itertools.groupby(sorted(data, key=key), key=key)

    folds = cast(List[Mapping[str, str]], [])
    for fold_id, fold_rows in groupby(raw_data, 'fold_id'):
        assert fold_id == len(folds)
        fold = cast(Dict[str, str], {})
        for row in fold_rows:
            fold[row['split_name']] = row['path']
        folds.append(fold)
    return folds


def generate_unique_id() -> str:
    return str(uuid.uuid1())


def simplify_trial(
    wiki: str,
    trial: Mapping,
    folds: Sequence[Mapping],
    parent_run_id: Optional[str]
) -> Mapping:
    """Simplify hyperopt trials into a storable format

    Pickling the trial object directly would be a pain to deal with
    for any downstream processing. Simplify it into a row conforming
    to mt.ModelParameters schema.
    """
    metrics = cast(List[Mapping], [])
    # We could have more metrics, but today only one
    for fold_id, fold_scores in enumerate(trial['result']['scores']):
        # fold_scores['metrics'] is the raw evals_result from xgboost
        for split_name, split_metrics in fold_scores['metrics'].items():
            for metric_name, iterations in split_metrics.items():
                for step, value in enumerate(iterations):
                    metrics.append(dict(
                        key=metric_name,
                        value=value,
                        step=step,
                        fold_id=fold_id,
                        split=split_name))

    return dict(
        run_id=generate_unique_id(),
        parent_run_id=parent_run_id,
        wikiid=wiki,
        started_at=trial['book_time'],
        completed_at=trial['refresh_time'],
        algorithm='xgboost',
        objective=trial['result']['params']['objective'],
        loss=trial['result']['loss'],
        params={str(k): str(v) for k, v in trial['result']['params'].items()},
        folds=folds,
        metrics=metrics,
        artifacts={}
    )


def tune_wiki(
    spark: SparkSession,
    folds: List[Mapping[str, str]],
    initial_num_trees: int,
    final_num_trees: int,
    iterations: int,
    num_observations: int,
    num_cv_jobs: Optional[int] = None,
) -> List[Mapping]:
    if num_cv_jobs is None:
        # default to running all cross validation in parallel, and
        # all hyperopt trials sequentially. Settings num_cv_jobs to
        # a multiple of folds will run multiple trials in parallel.
        num_cv_jobs = len(folds)
    results = mjolnir.training.xgboost.tune(
        folds, {'num_observations': num_observations},
        num_cv_jobs=num_cv_jobs,
        train_matrix='train',
        initial_num_trees=initial_num_trees,
        final_num_trees=final_num_trees,
        iterations=iterations,
        spark=spark)

    # Results contains a set of trials for each stage of tuning. We generally
    # only care about the final params and metrics for all evaluated models
    # so flatten it out.
    flat_trials = cast(List[Mapping], [])
    for trials in results['trials'].values():
        for trial in trials:
            flat_trials.append(trial)
    return flat_trials


def hyperparam(
    spark: SparkSession,
    date: str,
    output_path: str,
    output_table: str,

    training_files: Mapping,
    initial_num_trees: int,
    final_num_trees: int,
    iterations: int,
    num_cv_jobs: int,
    **kwargs
) -> DataFrame:
    # Select the xgboost formatted files and regroup to the expected
    # nested format.
    flat_folds = [row for row in training_files['rows']
                  if row['vec_format'] == 'xgboost' and row['split_name'] != 'all']
    folds = regroup_folds_metadata(flat_folds)

    trials = tune_wiki(
        spark, folds, initial_num_trees, final_num_trees,
        iterations, training_files['metadata']['num_obs'], num_cv_jobs)

    wiki = training_files['metadata']['wikiid']
    parent_run_id = generate_unique_id()
    results = [simplify_trial(wiki, trial, flat_folds, parent_run_id) for trial in trials]

    # The results at this point are not models, but evaluation metrics for
    # a variety of model parameters that were tested. An ideal set of parameters
    # can be chosen from this set and used to train against the full dataset.
    #
    # Again while not ideal to store tiny tables in spark...this keeps
    # consistency with our abstraction of everything being a transformation
    # from one dataframe to another.
    #
    # Each returned row represents a single set of parameters that was tested.
    # The "best" parameters can be determined by looking at the loss column,
    # or by more detailed inspection of the recorded metrics.
    return (
        spark.createDataFrame(results, mt.ModelParameters)  # type: ignore
        .repartition(1)
    )


def configure(parser: ArgumentParser) -> Callable:
    main = Cli('hyperparam', hyperparam, parser)
    # I/O
    main.require_training_files_partition()
    main.require_output_table(
        # Copy output partition spec from input
        lambda kwargs: kwargs['training_files']['partition_spec'],
        # Currently we always append the hyperparameter runs, the
        # old ones aren't really invalid and these new ones have new
        # id's. To be determined how later stages choose the right ones..
        mode='append')
    # tuning parameters
    main.add_argument('--initial-num-trees', type=int, required=True)
    main.add_argument('--final-num-trees', type=int, required=True)
    main.add_argument('--iterations', type=int, required=True)
    main.add_argument('--num-cv-jobs', type=int, default=None)
    return main
