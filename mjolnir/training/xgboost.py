from collections import defaultdict
from functools import partial
import tempfile
from typing import cast, Any, Callable, Dict, List, Mapping, Optional

import hyperopt
import numpy as np
from pyspark.sql import SparkSession
import xgboost as xgb

from mjolnir.training.tuning import make_cv_objective, ModelSelection
from mjolnir.utils import as_local_path, as_local_paths, as_output_file


def _coerce_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Force xgboost parameters into appropriate types

    The output from hyperopt is always floats, but some xgboost parameters
    explicitly require integers. Cast those as necessary

    Parameters
    ----------
    params : dict
        xgboost parameters

    Returns
    -------
    dict
        Input parameters coerced as necessary
    """
    def identity(x):
        return x

    def sloppy_int(x):
        try:
            return int(x)
        except ValueError:
            pass

        val = float(x)
        # This could fail for larger numbers due to fp precision, but not
        # expecting integer values larger than two digits here.
        if val.is_integer():
            return int(val)
        raise ValueError('Not parsable as integer: {}'.format(x))

    types = cast(Dict[str, Callable[[Any], Any]], defaultdict(lambda: identity))
    types.update({
        'max_depth': sloppy_int,
        'max_bin': sloppy_int,
        'num_class': sloppy_int,
        'silent': sloppy_int,
    })
    return {k: types[k](v) for k, v in params.items()}


def train(
    fold: Mapping[str, str],
    params: Mapping[str, Any],
    train_matrix: Optional[str] = None,
    spark: Optional[SparkSession] = None
) -> 'XGBoostModel':
    """Train a single xgboost ranking model.

    Primary entry point for hyperparameter tuning normalizes
    parameters and auto detects some values. Actual training is
    passed on to XGBoostModel.trainWithFiles

    Parameters
    ----------
    fold :
        Map from split name to data path. All provided splits will be
        evaluated on each boosting iteration.
    params :
        parameters to pass on to xgboost training
    train_matrix :
        Optional name of training matrix in fold. If not provided will
        auto-detect to either 'all' or 'train'
    spark:
        If provided, train remotely over spark

    Returns
    -------
    XGBoostModel
        Trained xgboost model
    """
    # hyperparameter tuning may have given us floats where we need
    # ints, so this gets all the types right for Java. Also makes
    # a copy of params so we don't modifying the incoming dict.
    # TODO: Does python care about this?  Even if not, in the end it seems
    # reasonable to not pass floats for integer values.
    params = _coerce_params(params)

    # TODO: Maybe num_rounds should just be external? But it's easier
    # to do hyperparameter optimization with a consistent dict interface
    kwargs = cast(Dict[str, Any], {
        'num_boost_round': 100,
    })

    if 'num_boost_round' in params:
        kwargs['num_boost_round'] = params['num_boost_round']
        del params['num_rounds']
    if 'early_stopping_rounds' in params:
        kwargs['early_stopping_rounds'] = params['early_stopping_rounds']
        del params['early_stopping_rounds']

    # Set some sane defaults for ranking tasks
    if 'objective' not in params:
        params['objective'] = 'rank:ndcg'
    if 'eval_metric' not in params:
        params['eval_metric'] = 'ndcg@10'
    # Not really ranking specific, but generally fastest method
    if 'tree_method' not in params:
        params['tree_method'] = 'hist'
    # Convenience for some situations, but typically be explicit
    # about the name of the matrix to train against.
    if train_matrix is None:
        train_matrix = "all" if "all" in fold else "train"

    if spark:
        return XGBoostModel.trainWithFilesRemote(spark, fold, train_matrix, params, **kwargs)
    else:
        return XGBoostModel.trainWithFiles(fold, train_matrix, params, **kwargs)


# Top level: matrix name
# Second level: metric name
# Inner list: stringified per-iteration metric value
EvalsResult = Mapping[str, Mapping[str, List[str]]]


class XGBoostBooster(object):
    """Wrapper for xgb.Booster usage in mjolnir

    Wraps the booster to distinguish what we have after training,
    the XGBoostModel, from what we write to disk, which is only the
    booster. Would be better if there was a clean way to wrap all
    the data up an serialize together, while working with xgboost's
    c++ methods that expect file paths.
    """
    def __init__(self, booster: xgb.Booster) -> None:
        self.booster = booster

    @staticmethod
    def loadBoosterFromHadoopFile(path: str) -> 'XGBoostBooster':
        with as_local_path(path) as local_path:
            return XGBoostBooster.loadBoosterFromLocalFile(local_path)

    @staticmethod
    def loadBoosterFromLocalFile(path: str) -> 'XGBoostBooster':
        booster = xgb.Booster.load_model(path)
        # TODO: Not having the training parameters or the evaluation metrics
        # almost makes this a different thing...
        return XGBoostBooster(booster)

    def saveBoosterAsHadoopFile(self, path: str):
        with as_output_file(path) as f:
            self.saveBoosterAsLocalFile(f.name)

    def saveBoosterAsLocalFile(self, path: str):
        # TODO: This doesn't save any metrics, should it?
        self.booster.save_model(path)


class XGBoostModel(XGBoostBooster):
    """xgboost booster along with train-time metrics

    TODO: Take XGBoostBooster as init arg instead of xgb.Booster?
    """

    def __init__(
        self,
        booster: xgb.Booster,
        evals_result: EvalsResult,
        params: Mapping[str, Any]
    ) -> None:
        super().__init__(booster)
        self.evals_result = evals_result
        self.params = params

    @staticmethod
    def trainWithFilesRemote(
        spark: SparkSession,
        fold: Mapping[str, str],
        train_matrix: str,
        params: Mapping[str, Any],
        **kwargs
    ) -> 'XGBoostModel':
        """Train model on a single remote spark executor.

        Silly hack to train models inside the yarn cluster. To train multiple
        models in parallel python threads will need to be used.  Wish pyspark
        had collectAsync.
        """
        nthread = int(spark.conf.get('spark.task.cpus', '1'))
        if 'nthread' not in params:
            params = dict(params, nthread=nthread)
        elif params['nthread'] != nthread:
            raise Exception("Executors have [{}] cpus but training requested [{}]".format(
                nthread, params['nthread']))

        return (
            spark.sparkContext
            .parallelize([1], 1)
            .map(lambda x: XGBoostModel.trainWithFiles(fold, train_matrix, params, **kwargs))
            .collect()[0]
        )

    @staticmethod
    def trainWithFiles(
        fold: Mapping[str, str],
        train_matrix: str,
        params: Mapping[str, Any],
        **kwargs
    ) -> 'XGBoostModel':
        """Wrapper around xgb.train

        This intentionally forwards to trainWithRDD, rather than
        trainWithDataFrame, as the underlying method currently prevents using
        rank:pairwise and metrics with @, such as ndcg@5.

        Parameters
        ----------
        fold :
            Map from split name to data path. All provided splits will be
            evaluated on each boosting iteration.
        train_matrix: str
            name of split in fold to train against
        params : dict
            XGBoost training parameters

        Returns
        -------
        mjolnir.training.xgboost.XGBoostModel
            trained xgboost ranking model
        """
        with as_local_paths(fold.values()) as local_paths:
            matrices = {name: xgb.DMatrix(path) for name, path in zip(fold.keys(), local_paths)}
            dtrain = matrices[train_matrix]
            evallist = [(dmat, name) for name, dmat in matrices.items()]
            metrics = cast(Mapping, {})
            booster = xgb.train(params, dtrain, evals=evallist, evals_result=metrics, **kwargs)
            return XGBoostModel(booster, metrics, params)

    def dump(self, features=None, with_stats=False, format="json"):
        """Dumps the xgboost model

        Parameters
        ----------
        features : list of str or None, optional
            list of features names, or None for no feature names in dump.
            (Default: None)
        withStats : bool, optional
            Should various additional statistics be included? These are not
            necessary for prediction. (Default: False)
        format : string, optional
            The format of dump to produce, either json or text. (Default: json)

        Returns
        -------
        str
            valid json string containing all trees
        """
        # Annoyingly the xgboost api doesn't take the feature map as a string, but
        # instead as a filename. Write the feature map out to a file if necessary.
        if features:
            feat_map = "\n".join(["%d %s q" % (i, fname) for i, fname in enumerate(features)])
            fmap_f = tempfile.NamedTemporaryFile(mode='w')
            fmap_f.write(feat_map)
            fmap_f.flush()
            fmap_path = fmap_f.name
        else:
            fmap_path = ''

        trees = self.booster.get_dump(fmap_path, with_stats, dump_format='json')
        # For whatever reason we get a json line per tree. Turn that into an array
        # so we have a single valid json string.
        return '[' + ','.join(trees) + ']'


def cv_transformer(model: XGBoostModel, params: Mapping[str, Any]):
    """Report model metrics in format expected by model selection"""
    metric = params['eval_metric']
    return {
        'train': model.evals_result['train'][metric][-1],
        'test': model.evals_result['test'][metric][-1],
        'metrics': model.evals_result,
    }


def tune(
    folds: List[Mapping[str, str]],
    stats: Dict,
    train_matrix: str,
    num_cv_jobs: int = 5,
    initial_num_trees: int = 100,
    final_num_trees: int = 500,
    iterations: int = 150,
    spark: Optional[SparkSession] = None
):
    """Find appropriate hyperparameters for training df

    This is far from perfect, hyperparameter tuning is a bit of a black art
    and could probably benefit from human interaction at each stage. Various
    parameters depend a good bit on the number of samples in df, and how
    that data is shaped.

    Below is tuned for a dataframe with approximatly 10k normalized queries,
    110k total queries, and 2.2M samples. This is actually a relatively small
    dataset, we should rework the values used with larger data sets if they
    are promising. It may also be that the current feature space can't take
    advantage of more samples.

    Note that hyperopt uses the first 20 iterations to initialize, during those
    first 20 this is a strictly random search.

    Parameters
    ----------
    folds : list of dict containing train and test keys
    stats : dict
        stats about the fold from the make_folds utility script
    num_cv_jobs : int, optional
        The number of cross validation folds to train in parallel. (Default: 5)
    initial_num_trees: int, optional
        The number of trees to do most of the hyperparameter tuning with. This
        should be large enough to be resonably representative of the final
        training size. (Default: 100)
    final_num_trees: int, optional
        The number of trees to do the final eta optimization with. If set to
        None the final eta optimization will be skipped and initial_n_tree will
        be kept.

    Returns
    -------
    dict
        Dict with two keys, trials and params. params is the optimal set of
        parameters. trials contains a dict of individual optimization steps
        performed, each containing a hyperopt.Trials object recording what
        happened.
    """
    num_obs = stats['num_observations']

    if num_obs > 8000000:
        dataset_size = 'xlarge'
    elif num_obs > 1000000:
        dataset_size = 'large'
    elif num_obs > 500000:
        dataset_size = 'med'
    elif num_obs > 500:
        dataset_size = 'small'
    else:
        dataset_size = 'xsmall'

    # Setup different tuning profiles for different sizes of datasets.
    tune_spaces = [
        ('initial', {
            'iterations': iterations,
            'space': {
                'xlarge': {
                    'eta': hyperopt.hp.uniform('eta', 0.3, 0.8),
                    # Have seen values of 7 and 10 as best on roughly same size
                    # datasets from different wikis. It really just depends.
                    'max_depth': hyperopt.hp.quniform('max_depth', 6, 11, 1),
                    'min_child_weight': hyperopt.hp.qloguniform(
                        'min_child_weight', np.log(10), np.log(500), 10),
                    # % of features to use for each tree. helps prevent overfit
                    'colsample_bytree': hyperopt.hp.quniform('colsample_bytree', 0.8, 1, .01),
                    'subsample': hyperopt.hp.quniform('subsample', 0.8, 1, .01),
                },
                'large': {
                    'eta': hyperopt.hp.uniform('eta', 0.3, 0.6),
                    'max_depth': hyperopt.hp.quniform('max_depth', 5, 9, 1),
                    'min_child_weight': hyperopt.hp.qloguniform(
                        'min_child_weight', np.log(10), np.log(300), 10),
                    'colsample_bytree': hyperopt.hp.quniform('colsample_bytree', 0.8, 1, .01),
                    'subsample': hyperopt.hp.quniform('subsample', 0.8, 1, .01),
                },
                'med': {
                    'eta': hyperopt.hp.uniform('eta', 0.1, 0.6),
                    'max_depth': hyperopt.hp.quniform('max_depth', 4, 7, 1),
                    'min_child_weight': hyperopt.hp.qloguniform(
                        'min_child_weight', np.log(10), np.log(300), 10),
                    'colsample_bytree': hyperopt.hp.quniform('colsample_bytree', 0.8, 1, .01),
                    'subsample': hyperopt.hp.quniform('subsample', 0.8, 1, .01),
                },
                'small': {
                    'eta': hyperopt.hp.uniform('eta', 0.1, 0.4),
                    'max_depth': hyperopt.hp.quniform('max_depth', 3, 6, 1),
                    'min_child_weight': hyperopt.hp.qloguniform(
                        'min_child_weight', np.log(10), np.log(100), 10),
                    'colsample_bytree': hyperopt.hp.quniform('colsample_bytree', 0.8, 1, .01),
                    'subsample': hyperopt.hp.quniform('subsample', 0.8, 1, .01),
                },
                'xsmall': {
                    'eta': hyperopt.hp.uniform('eta', 0.1, 0.4),
                    'max_depth': hyperopt.hp.quniform('max_depth', 3, 6, 1),
                    # Never use for real data, but convenient for tiny sets in test suite
                    'min_child_weight': 0,
                    'colsample_bytree': hyperopt.hp.quniform('colsample_bytree', 0.8, 1, .01),
                    'subsample': hyperopt.hp.quniform('subsample', 0.8, 1, .01),
                }
            }[dataset_size]
        })
    ]

    if final_num_trees is not None and final_num_trees != initial_num_trees:
        tune_spaces.append(('trees', {
            'iterations': 30,
            'space': {
                'num_rounds': final_num_trees,
                'eta': hyperopt.hp.uniform('eta', 0.1, 0.4),
            }
        }))

    # Baseline parameters to start with. Roughly tuned by what has worked in
    # the past. These vary though depending on number of training samples. These
    # defaults are for the smallest of wikis, which are then overridden for larger
    # wikis
    space = {
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg@10',
        'num_rounds': initial_num_trees,
        'min_child_weight': 200,
        'max_depth': {
            'xlarge': 7,
            'large': 6,
            'med': 5,
            'small': 4,
            'xsmall': 3,
        }[dataset_size],
        'gamma': 0,
        'subsample': 1.0,
        'colsample_bytree': 0.8,
    }

    tuner = ModelSelection(space, tune_spaces)
    train_func = make_cv_objective(
        partial(train, spark=spark), folds, num_cv_jobs,
        cv_transformer, train_matrix=train_matrix)
    trials_pool = tuner.build_pool(folds, num_cv_jobs)
    return tuner(train_func, trials_pool)
