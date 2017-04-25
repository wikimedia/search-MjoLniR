import mjolnir.spark
import pyspark.sql
from pyspark.sql import functions as F
import tempfile

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
