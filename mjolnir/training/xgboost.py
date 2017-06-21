import hyperopt
import mjolnir.spark
import mjolnir.training.tuning
import numpy as np
import pprint
import pyspark.sql
from pyspark.sql import functions as F
import tempfile
import scipy.sparse

# Example Command line:
# PYSPARK_PYTHON=venv/bin/python SPARK_CONF_DIR=/etc/spark/conf ~/spark-2.1.0-bin-hadoop2.6/bin/pyspark \
#     --master yarn \
#     --jars ~/mjolnir_2.11-1.0.jar \
#     --driver-class-path ~/mjolnir_2.11-1.0.jar \
#     --archives 'mjolnir_venv.zip#venv' \
#     --files /usr/lib/libhdfs.so.0.0.0 \
#     --executor-cores 4 \
#     --executor-memory 4G \
#     --conf spark.dynamicAllocation.maxExecutors=40 \
#     --conf spark.task.cpus=4


def prep_training(df, num_partitions=None):
    """Prepare a dataframe for training

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
        Dataframe repartitioned and sorted into query groups.
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
        .sortWithinPartitions('queryId'))

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


def train(df, params, num_workers=None):
    """Train a single xgboost ranking model.

    df : pyspark.sql.DataFrame
        Training data
    params : dict
        parameters to pass on to xgboost
    num_workers : int, optional
        The number of executors to train with. If not provided then
        'groupData' *must* be present in params and num_workers will
        be set to the number of partitions in df.

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
    num_rounds = params['num_rounds']
    del params['num_rounds']
    # Set some sane defaults for ranking tasks
    if 'objective' not in params:
        params['objective'] = 'rank:ndcg'
    if 'eval_metric' not in params:
        params['eval_metric'] = 'ndcg@10'

    if num_workers is None:
        num_workers = df.rdd.getNumPartitions()
        if 'groupData' in params:
            assert params['groupData'].length() == num_workers
            df_grouped = df
        else:
            df_grouped, j_groups = prep_training(df, num_workers)
            params['groupData'] = j_groups
    elif 'groupData' in params:
        df_grouped = df
    else:
        df_grouped, j_groups = prep_training(df, num_workers)
        params['groupData'] = j_groups

    # We must have the same number of partitions here as workers the model will
    # be trained with, or xgboost4j-spark will repartition and the c++ library
    # will throw an exception. It's much cleaner to fail-fast here rather than
    # figuring out c++ errors through JNI from remote workers.
    assert df_grouped.rdd.getNumPartitions() == num_workers
    assert 'groupData' in params
    assert params['groupData'].length() == num_workers

    return XGBoostModel.trainWithDataFrame(df_grouped, params, num_rounds,
                                           num_workers, feature_col='features',
                                           label_col='label')


class XGBoostModel(object):
    def __init__(self, j_xgb_model):
        self._j_xgb_model = j_xgb_model

    @staticmethod
    def trainWithDataFrame(trainingData, params, num_rounds, num_workers, objective=None,
                           eval_metric=None, use_external_memory=False, missing=float('nan'),
                           feature_col='features', label_col='label'):
        """Wrapper around scala XGBoostModel.trainWithRDD

        This intentionally forwards to trainWithRDD, rather than
        trainWithDataFrame, as the underlying method currently prevents using
        rank:pairwise and metrics with @, such as ndcg@5.

        Parameters
        ----------
        trainingData : pyspark.sql.DataFrame
        params : dict
        num_rounds : int
        num_workers : int
        objective : py4j.java_gateway.JavaObject, optional
            Allows providing custom objective implementation. (Default: None)
        eval_metric : py4j.java_gateway.JavaObject, optional
            Allows providing a custom evaluation metric implementation.
            (Default: None)
        use_external_memory : bool, optional
            indicate whether to use external memory cache, by setting this flag
            as true, the user may save the RAM cost for running XGBoost within
            spark.  Essentially this puts the data on local disk, and takes
            advantage of the kernel disk cache(maybe?). (Default: False)
        missing : float, optional
            The value representing the missing value in the dataset. features with
            this value will be removed and the vectors treated as sparse. (Default: nan)
        feature_col : string, optional
            The dataframe column holding feature vectors. (Default: features)
        label_col : string, optional
            The dataframe column holding labels. (Default: label)

        Returns
        -------
        mjolnir.training.xgboost.XGBoostModel
            trained xgboost ranking model
        """
        sc = trainingData._sc
        j_params = sc._jvm.scala.collection.immutable.HashMap()
        for k, v in params.items():
            j_params = j_params.updated(k, v)

        j_rdd = sc._jvm.org.wikimedia.search.mjolnir.PythonUtils.toLabeledPoints(
            trainingData._jdf, feature_col, label_col)

        j_xgb_model = sc._jvm.ml.dmlc.xgboost4j.scala.spark.XGBoost.trainWithRDD(
            j_rdd, j_params, num_rounds, num_workers, objective, eval_metric,
            use_external_memory, missing)
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

    def dump(self, feature_map=None, with_stats=False, format="json"):
        """Dumps the xgboost model

        Parameters
        ----------
        featureMap : str or None, optional
            Formatted as per xgboost documentation for featmap.txt.
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
        if feature_map is None:
            fmap_path = None
        else:
            fmap_f = tempfile.NamedTemporaryFile()
            fmap_f.write(feature_map)
            fmap_path = fmap_f.name
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

    def saveModelAsHadoopFile(self, sc, path):
        j_sc = sc._jvm.org.apache.spark.api.java.JavaSparkContext.toSparkContext(sc._jsc)
        self._j_xgb_model.saveModelAsHadoopFile(path, j_sc)

    @staticmethod
    def loadModelFromHadoopFile(sc, path):
        j_sc = sc._jvm.org.apache.spark.api.java.JavaSparkContext.toSparkContext(sc._jsc)
        j_xgb_model = sc._jvm.ml.dmlc.xgboost4j.scala.spark.XGBoost.loadModelFromHadoopFile(
            path, j_sc)
        return XGBoostModel(j_xgb_model)


# from https://gist.github.com/hernamesbarbara/7238736
def _loess_predict(X, y_tr, X_pred, bandwidth):
    X_tr = np.column_stack((np.ones_like(X), X))
    X_te = np.column_stack((np.ones_like(X_pred), X_pred))
    y_te = []
    for x in X_te:
        ws = np.exp(-np.sum((X_tr - x)**2, axis=1) / (2 * bandwidth**2))
        W = scipy.sparse.dia_matrix((ws, 0), shape=(X_tr.shape[0],) * 2)
        theta = np.linalg.pinv(X_tr.T.dot(W.dot(X_tr))).dot(X_tr.T.dot(W.dot(y_tr)))
        y_te.append(np.dot(x, theta))
    return np.array(y_te)


def _estimate_best_eta(trials, source_etas, length=1e4):
    """Estimate the best eta from a small sample of trials

    The final stage of training can take quite some time, at 10 to 15 minutes
    or more per model evaluated. The relationship between eta and ndcg@10 along
    with eta and true loss is fairly stable, so instead of searching the whole space
    evaluate a few evenly spaced points and then try to fit a line to determine
    the best eta.

    Best eta is chosen by finding where the derivative of ndcg@10 vs true loss first
    transitions from >1 to <=1

    Parameters
    ----------
    trials : hyperopt.Trials
        Trials object that was used to tune only eta
    source_etas : list of float
        For some reason hyperopt.hp.choice doesn't include the actual value in
        the trials object, only the index. The results as indexed into this
        list to get the actual eta tested.
    length : int
        Number of eta points to estimate

    Returns
    -------
    float
        Estimated best ETA
    """

    ndcg10 = np.asarray([-l for l in trials.losses()])
    true_loss = np.asarray([r.get('true_loss') for r in trials.results])
    eta = np.asarray([source_etas[v] for v in trials.vals['eta']])

    # Range of predictions we want to make
    eta_pred = np.arange(np.min(eta), np.max(eta), (np.max(eta) - np.min(eta)) / length)
    # Predicted ndcg@10 values for eta_pred
    # TODO: Can 0.02 not be magic? Was chosen by hand on one sample
    ndcg10_pred = _loess_predict(eta, ndcg10, eta_pred, 0.02)
    # Predicted true loss for eta_pred
    # TODO: Can 0.03 not be magic? Was chosen by hand on one sample
    true_loss_pred = _loess_predict(eta, true_loss, eta_pred, 0.03)

    # Find the first point where derivative transitions from > 1 to <= 1.
    # TODO: What if the sample is from too narrow a range, and doesn't capture this?
    derivative = np.diff(ndcg10_pred) / np.diff(true_loss_pred)
    idx = (np.abs(derivative-1)).argmin()

    # eta for point closest to transition of derivative from >1 to <=1>
    return eta_pred[idx]


def tune(df, num_folds=5, num_fold_partitions=100, num_cv_jobs=5, num_workers=5,
         target_node_evaluations=5000):
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
    df : pyspark.sql.DataFrame
    num_folds : int, optional
        The number of cross validation folds to use while tuning. (Default: 5)
    num_fold_partitions : int, optional
        The number of partitions to use when calculating folds. For small
        datasets this needs to be a reasonably small number. For medium to
        large the default is reasonable. (Default: 100)
    num_cv_jobs : int, optional
        The number of cross validation folds to train in parallel. (Default: 5)
    num_workers : int, optional
        The number of spark executors to use per fold for training. The total
        number of executors used will be (num_cv_jobs * num_workers). Generally
        prefer executors with more cpu's over a higher number of workers where
        possible. (Default: 5)
    target_node_evaluations : int, optional
        The approximate number of node evaluations per prediction that the
        final result will require. This controls the number of trees used in
        the final result. The number of trees will be (target_node_evaluations
        / optimal_max_depth). This is by far the most expensive part to tune,
        setting to None skips this and uses a constant 100 trees.
        (Default: 5000)

    Returns
    -------
    dict
        Dict with two keys, trials and params. params is the optimal set of
        parameters. trials contains a dict of individual optimization steps
        performed, each containing a hyperopt.Trials object recording what
        happened.
    """
    def eval_space(space, max_evals):
        """Eval a space using standard hyperopt"""
        best, trials = mjolnir.training.tuning.hyperopt(
            df, train, space, max_evals=max_evals,
            num_folds=num_folds, num_fold_partitions=num_fold_partitions,
            num_cv_jobs=num_cv_jobs, num_workers=num_workers)
        for k, v in space.items():
            if not np.isscalar(v):
                print 'best %s: %f' % (k, best[k])
        return best, trials

    def eval_space_grid(space):
        """Eval all points in the space via a grid search"""
        best, trials = mjolnir.training.tuning.grid_search(
            df, train, space, num_folds=num_folds, num_fold_partitions=num_fold_partitions,
            num_cv_jobs=num_cv_jobs, num_workers=num_workers)
        for k, v in space.items():
            if not np.isscalar(v):
                print 'best %s: %f' % (k, best[k])
        return best, trials

    # Baseline parameters to start with. Roughly tuned by what has worked in
    # the past. These vary though depending on number of training samples
    space = {
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg@10',
        'num_rounds': 100,
        'min_child_weight': 200,
        'max_depth': 6,
        'gamma': 0,
        'subsample': 1.0,
        'colsample_bytree': 0.8,
    }

    # Find an eta that gives good results with only 100 trees. This is done
    # so most of the tuning is relatively quick. A final step will re-tune
    # eta with more trees.
    etas = np.linspace(0.3, 0.7, 30)
    space['eta'] = hyperopt.hp.choice('eta', etas)
    best_eta, trials_eta = eval_space_grid(space)
    space['eta'] = _estimate_best_eta(trials_eta, etas)
    pprint.pprint(space)

    # Determines the size of each tree. Larger trees increase model complexity
    # and make it more likely to overfit the training data. Larger trees also
    # do a better job at capturing interactions between features. Larger training
    # sets support deeper trees. Not all trees will be this depth, min_child_weight
    # gamma, and regularization all push back on this.
    space['max_depth'] = hyperopt.hp.quniform('max_depth', 4, 10, 1)
    # The minimum number of samples that must be in each leaf node. This pushes
    # back against tree depth, preventing the tree from growing if a potential
    # split applies to too few samples. ndcg@10 on the test set increases linearly
    # with smaller min_child_weight, but true_loss also increases.
    space['min_child_weight'] = hyperopt.hp.qloguniform('min_child_weight', np.log(10), np.log(2000), 10)

    # TODO: Somewhat similar to eta, as min_child_weight decreases the
    # true_loss increases. Need to figure out how to choose the max_depth that
    # provides best ndcg@10 without losing generalizability.
    best_complexity, trials_complexity = eval_space(space, 50)
    space['max_depth'] = int(best_complexity['max_depth'])
    space['min_child_weight'] = int(best_complexity['min_child_weight'])
    pprint.pprint(space)

    # Gamma also controls complexity, but in a less brute force manner than min_child_weight.
    # Essentially each newly generated split has a gain value calculated indicating how
    # good the split is. gamma is the minimum gain a split must achieve to be considered
    # a good split. Gamma has a mostly linear relationship with true_loss. The relationship
    # to ndcg@10 is less clear, although documentation suggests with enough data points
    # gamma vs loss should draw a U shaped graph (inverted in case of ndcg@10)
    # TODO: Should we even tune this? The results are all over the place, suggesting we may simply
    # be finding some gamma that matches the CV folds best and not something generalizable.
    space['gamma'] = hyperopt.hp.quniform('gamma', 0, 3, 0.01)
    best_gamma, trials_gamma = eval_space(space, 50)
    space['gamma'] = best_gamma['gamma']
    pprint.pprint(space)

    # subsample helps make the model more robust to noisy data. For each update to
    # a tree only this % of samples are considered.
    space['subsample'] = hyperopt.hp.quniform('subsample', 0.8, 1, .01)
    # colsample also helps make the model more robust to noise. For each update
    # to a tree only this % of features are considered.
    space['colsample_bytree'] = hyperopt.hp.quniform('colsample_bytree', 0.8, 1, .01)

    # With a high min_child_weight subsampling of any kind gives a linear decrease
    # in quality. But with a relatively low min_child_weight it can give some benefits,
    # pushing back against over fitting due to small amounts of data per leaf.
    # colsample is less clear, with 0.8 and 1.0 having similar results.
    best_noise, trials_noise = eval_space(space, 50)
    space['subsample'] = best_noise['subsample']
    space['colsample_bytree'] = best_noise['colsample_bytree']
    pprint.pprint(space)

    # Finally increase the number of trees to our target, which is mostly based
    # on how computationally expensive it is to generate predictions with the final
    # model. Find the optimal eta for this new # of trees. This step can take as
    # much time as all previous steps combined, and then some, so it can be disabled
    # with target_node_evalations of None.
    if target_node_evaluations is None:
        trials_trees = None
        trials_final = trials_noise
    else:
        space['num_rounds'] = target_node_evaluations / space['max_depth']
        # TODO: Is 30 steps right amount? too many? too few? This generally
        # uses a large number of trees which takes 10 to 20 minutes per evaluation.
        # That means evaluating 15 points is 2.5 to 5 hours.
        etas = np.linspace(0.01, 0.3, 30)
        space['eta'] = hyperopt.hp.choice('eta', etas)
        best_trees, trials_trees = eval_space_grid(space)
        trials_final = trials_trees
        space['eta'] = _estimate_best_eta(trials_trees, etas)
        pprint.pprint(space)

    best_trial = np.argmin(trials_final.losses())
    loss = trials_final.losses()[best_trial]
    true_loss = trials_final.results[best_trial].get('true_loss')

    return {
        'trials': {
            'initial': trials_eta,
            'complexity': trials_complexity,
            'gamma': trials_gamma,
            'noise': trials_noise,
            'trees': trials_trees,
        },
        'params': space,
        'metrics': {
            'test': -loss,
            'train': -loss + true_loss
        }
    }
