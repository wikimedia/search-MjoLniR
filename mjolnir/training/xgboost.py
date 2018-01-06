from __future__ import absolute_import
import functools
import hyperopt
import math
import mjolnir.spark
import mjolnir.training.hyperopt
from multiprocessing.dummy import Pool
import numpy as np
import pyspark
import pyspark.sql
from pyspark.sql import functions as F
import tempfile


def prep_training(df, num_partitions=None):
    """Prepare a dataframe for training

    This is no longer used for training. It can stlil be used to run predictions
    or evaluations on a model. Training uses make_fold utility now.

    Ranking models in XGBoost require rows for the same query to be provided
    consequtively within a single partition. It additionally requires a
    groupData parameter that indicates the number of consequtive items per-query
    per-partition. The resulting dataframe must *not* be changed between here
    and training/evaluation or there is a risk of invalidating the groupData.
    Additionally when training the model xgboost4j-spark must be provided a
    number of workers equal to the number of partitions used here, or it will
    repartition the data and invalidate the groupData.

    Repartition by unique queries to bring all rows for a single query within a
    single partition, sort within partitions to make the rows for same query
    consecutive, and then count the number of consequtive items.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame to be trained/evaluated with xgboost
    num_partitions : int
        Number of partitions to create. This must be equal to the number of
        executors that will be used to train a model. For model evaluation this
        can be anything. If none then the number of partitions will match df.
        (Default: None)

    Returns
    -------
    pyspark.sql.DataFrame
        Dataframe repartitioned and sorted into query groups. The dataframe is
        additionally cached and must be unpersisted when no longer necessary
    py4j.java_gateway.JavaObject
        group information for xgboost groupData parameter. scala type
        is Seq[Seq[Int]].
    """
    mjolnir.spark.assert_columns(df, ['label', 'features', 'wikiid', 'query'])

    if num_partitions is None:
        num_partitions = df.rdd.getNumPartitions()

    df_grouped = (
        # TODO: Should probably create queryId and normQueryId columns early in
        # the pipeline so various tasks can accept a single column to work with.
        df.select('label', 'features', F.concat('wikiid', 'query').alias('queryId'))
        .repartition(num_partitions, 'queryId')
        .sortWithinPartitions('queryId')
        .cache())

    j_groups = df._sc._jvm.org.wikimedia.search.mjolnir.PythonUtils.calcQueryGroups(
        df_grouped._jdf, 'queryId')

    return df_grouped, j_groups


def _coerce_params(params):
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

    types = {
        'max_depth': int,
        'max_bin': int,
        'num_class': int,
        'silent': int,
    }
    retval = params.copy()
    for (k, val_type) in types.items():
        if k in params:
            retval[k] = val_type(params[k])
    return retval


def train(fold, params, train_matrix=None):
    """Train a single xgboost ranking model.

    fold: dict
        map from split names to list of data partitions
    params : dict
        parameters to pass on to xgboost

    Returns
    -------
    XGBoostModel
        Trained xgboost model
    """
    # hyperparameter tuning may have given us floats where we need
    # ints, so this gets all the types right for Java. Also makes
    # a copy of params so we don't modifying the incoming dict.
    params = _coerce_params(params)
    # TODO: Maybe num_rounds should just be external? But it's easier
    # to do hyperparameter optimization with a consistent dict interface
    kwargs = {
        'num_rounds': 100,
        'early_stopping_round': 0,
    }
    if 'num_rounds' in params:
        kwargs['num_rounds'] = params['num_rounds']
        del params['num_rounds']
    if 'early_stopping_round' in params:
        kwargs['early_stopping_round'] = params['early_stopping_round']
        del params['early_stopping_round']

    # Set some sane defaults for ranking tasks
    if 'objective' not in params:
        params['objective'] = 'rank:ndcg'
    if 'eval_metric' not in params:
        params['eval_metric'] = 'ndcg@10'

    # Convenience for some situations, but typically be explicit
    # about the name of the matrix to train against.
    if train_matrix is None:
        train_matrix = "all" if "all" in fold else "train"

    return XGBoostModel.trainWithFiles(fold, train_matrix, params, **kwargs)


class XGBoostSummary(object):
    def __init__(self, j_xgb_summary):
        self._j_xgb_summary = j_xgb_summary

    def train(self):
        return list(self._j_xgb_summary.trainObjectiveHistory())

    def test(self):
        if self._j_xgb_summary.testObjectiveHistory().isEmpty():
            return None
        else:
            return list(self._j_xgb_summary.testObjectiveHistory().get())


class XGBoostModel(object):
    def __init__(self, j_xgb_model):
        self._j_xgb_model = j_xgb_model

    @staticmethod
    def trainWithFiles(fold, train_matrix, params, num_rounds=100,
                       early_stopping_round=0):
        """Wrapper around scala XGBoostModel.trainWithRDD

        This intentionally forwards to trainWithRDD, rather than
        trainWithDataFrame, as the underlying method currently prevents using
        rank:pairwise and metrics with @, such as ndcg@5.

        Parameters
        ----------
        fold: dict
            map from string name to list of data files for the split
        train_matrix: str
            name of split in fold to train against
        params : dict
            XGBoost training parameters
        num_rounds : int
            Maximum number of boosting rounds to perform
        early_stopping_round : int, optional
            Quit training after this many rounds with no improvement in
            test set eval. 0 disables behaviour. (Default: 0)

        Returns
        -------
        mjolnir.training.xgboost.XGBoostModel
            trained xgboost ranking model
        """
        sc = pyspark.SparkContext.getOrCreate()
        # Type is Seq[Map[String, String]]
        j_fold = sc._jvm.PythonUtils.toSeq([sc._jvm.PythonUtils.toScalaMap(x) for x in fold])
        # Type is Map[String, Any]
        j_params = sc._jvm.scala.collection.immutable.HashMap()
        for k, v in params.items():
            j_params = j_params.updated(k, v)

        j_xgb_model = sc._jvm.org.wikimedia.search.mjolnir.MlrXGBoost.trainWithFiles(
            sc._jsc, j_fold, train_matrix, j_params, num_rounds,
            early_stopping_round)
        return XGBoostModel(j_xgb_model)

    def transform(self, df_test):
        """Generate predictions and attach to returned df_test

        Parameters
        ----------
        df_test : pyspark.sql.DataFrame
            A dataframe containing feature vectors to run predictions
            against. The features must use the same column name as used
            when training.

        Returns
        -------
        pyspark.sql.DataFrame
            The original dataframe with an additional 'prediction' column
        """
        j_df = self._j_xgb_model.transform(df_test._jdf)
        return pyspark.sql.DataFrame(j_df, df_test.sql_ctx)

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
            fmap_f = tempfile.NamedTemporaryFile()
            fmap_f.write(feat_map)
            fmap_f.flush()
            fmap_path = fmap_f.name
        else:
            fmap_path = None
        # returns an Array[String] from scala, where each element of the array
        # is a json string representing a single tree.
        j_dump = self._j_xgb_model.booster().getModelDump(fmap_path, with_stats, format)
        return '[' + ','.join(list(j_dump)) + ']'

    def eval(self, df_test, j_groups=None, feature_col='features', label_col='label'):
        """Evaluate the model against a dataframe

        Evaluates the model using the eval_metric that was provided when
        training the model.

        Parameters
        ----------
        df_test : pyspark.sql.DataFrame
            A dataframe containing feature vectors and labels to evaluate
            prediction accuracy against.
        j_groups : py4j.java_gateway.JavaObject, optional
            A Seq[Seq[Int]] indicating the groups (queries) within the
            dataframe partitions. If not provided df_test will be repartitioned
            and sorted so this can be calculated. (Default: None)
        feature_col : string, optional
            The dataframe column holding feature vectors. (Default: features)
        label_col : string, optional
            The dataframe column holding labels. (Default: label)

        Returns
        -------
        float
            Metric representing the prediction accuracy of the model.
        """
        if j_groups is None:
            num_partitions = df_test.rdd.getNumPartitions()
            df_grouped, j_groups = prep_training(df_test, num_partitions)
        else:
            assert df_test.rdd.getNumPartitions() == j_groups.length()
            df_grouped = df_test

        j_rdd = df_test._sc._jvm.org.wikimedia.search.mjolnir.PythonUtils.toLabeledPoints(
            df_grouped._jdf, feature_col, label_col)
        score = self._j_xgb_model.eval(j_rdd, 'test', None, 0, False, j_groups)
        return float(score.split('=')[1].strip())

    def summary(self):
        return XGBoostSummary(self._j_xgb_model.summary())

    def saveModelAsHadoopFile(self, sc, path):
        j_sc = sc._jvm.org.apache.spark.api.java.JavaSparkContext.toSparkContext(sc._jsc)
        self._j_xgb_model.saveModelAsHadoopFile(path, j_sc)

    def saveModelAsLocalFile(self, path):
        self._j_xgb_model.booster().saveModel(path)

    @staticmethod
    def loadModelFromHadoopFile(sc, path):
        j_sc = sc._jvm.org.apache.spark.api.java.JavaSparkContext.toSparkContext(sc._jsc)
        j_xgb_model = sc._jvm.ml.dmlc.xgboost4j.scala.spark.XGBoost.loadModelFromHadoopFile(
            path, j_sc)
        return XGBoostModel(j_xgb_model)

    @staticmethod
    def loadModelFromLocalFile(sc, path):
        j_xgb_booster = sc._jvm.ml.dmlc.xgboost4j.scala.XGBoost.loadModel(path)
        j_xgb_model = sc._jvm.ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel(j_xgb_booster)
        return XGBoostModel(j_xgb_model)


def tune(folds, stats, train_matrix, num_cv_jobs=5, initial_num_trees=100, final_num_trees=500):
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
    cv_pool = None
    if num_cv_jobs > 1:
        cv_pool = Pool(num_cv_jobs)

    # Configure the trials pool large enough to keep cv_pool full
    num_folds = len(folds)
    num_workers = len(folds[0])
    trials_pool_size = int(math.floor(num_cv_jobs / (num_folds * num_workers)))
    if trials_pool_size > 1:
        print 'Running %d cross validations in parallel' % (trials_pool_size)
        trials_pool = Pool(trials_pool_size)
    else:
        trials_pool = None

    train_func = functools.partial(train, train_matrix=train_matrix)

    def eval_space(space, max_evals):
        """Eval a space using standard hyperopt"""
        best, trials = mjolnir.training.hyperopt.minimize(
            folds, train_func, space, max_evals=max_evals,
            cv_pool=cv_pool, trials_pool=trials_pool)
        for k, v in space.items():
            if not np.isscalar(v):
                print 'best %s: %f' % (k, best[k])
        return best, trials

    num_obs = stats['num_observations']

    if num_obs > 8000000:
        dataset_size = 'xlarge'
    elif num_obs > 1000000:
        dataset_size = 'large'
    elif num_obs > 500000:
        dataset_size = 'med'
    else:
        dataset_size = 'small'

    # Setup different tuning profiles for different sizes of datasets.
    tune_spaces = [
        ('initial', {
            'iterations': 150,
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
                },
                'large': {
                    'eta': hyperopt.hp.uniform('eta', 0.3, 0.6),
                    'max_depth': hyperopt.hp.quniform('max_depth', 5, 9, 1),
                    'min_child_weight': hyperopt.hp.qloguniform(
                        'min_child_weight', np.log(10), np.log(300), 10),
                    'colsample_bytree': hyperopt.hp.quniform('colsample_bytree', 0.8, 1, .01),
                },
                'med': {
                    'eta': hyperopt.hp.uniform('eta', 0.1, 0.6),
                    'max_depth': hyperopt.hp.quniform('max_depth', 4, 7, 1),
                    'min_child_weight': hyperopt.hp.qloguniform(
                        'min_child_weight', np.log(10), np.log(300), 10),
                    'colsample_bytree': hyperopt.hp.quniform('colsample_bytree', 0.8, 1, .01),
                },
                'small': {
                    'eta': hyperopt.hp.uniform('eta', 0.1, 0.4),
                    'max_depth': hyperopt.hp.quniform('max_depth', 3, 6, 1),
                    'min_child_weight': hyperopt.hp.qloguniform(
                        'min_child_weight', np.log(10), np.log(100), 10),
                    'colsample_bytree': hyperopt.hp.quniform('colsample_bytree', 0.8, 1, .01),
                }
            }[dataset_size]
        }),
        ('trees', {
            'iterations': 30,
            'condition': lambda: final_num_trees is not None and final_num_trees != initial_num_trees,
            'space': {
                'num_rounds': final_num_trees,
                'eta': hyperopt.hp.uniform('eta', 0.1, 0.4),
            }
        })
    ]

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
        }[dataset_size],
        'gamma': 0,
        'subsample': 1.0,
        'colsample_bytree': 0.8,
    }

    stages = []
    for name, stage_params in tune_spaces:
        if 'condition' in stage_params and not stage_params['condition']():
            continue
        tune_space = stage_params['space']
        for name, dist in tune_space.items():
            space[name] = dist
        best, trials = eval_space(space, stage_params['iterations'])
        for name in tune_space.keys():
            space[name] = best[name]
        stages.append((name, trials))

    trials = stages[-1][1]
    best_trial = np.argmin(trials.losses())
    loss = trials.losses()[best_trial]
    true_loss = trials.results[best_trial].get('true_loss')

    return {
        'trials': {
            'initial': trials,
        },
        'params': space,
        'metrics': {
            'cv-test': -loss,
            'cv-train': -loss + true_loss
        }
    }
