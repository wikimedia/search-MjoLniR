"""
Support for making test/train or k-fold splits
"""
from __future__ import absolute_import
from collections import defaultdict
import mjolnir.spark
import py4j.protocol
from pyspark.sql import functions as F


def split(df, splits, output_column='fold', num_partitions=100):
    """Assign splits to a dataframe of search results

    Individual hits from the same normalized query are not independent,
    they should have large overlaps in result sets and relevance labels,
    so the splitting happens at the normalized query level.

    Although the splitting is happening at the normalized query level, the
    split percentage is still with respect to the number of rows assigned to
    each split, not the number of normalized queries. This additionally ensures
    that the split is equal per wiki, meaning an 80/20 split will result in
    an 80/20 split for each wiki.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input data frame containing (wikiid, norm_query_id) columns. If this is
        expensive to compute it should be cached, as it will be used twice.
    splits: list
        List of percentages, summing to 1, to split the input dataframe
        into.
    output_column : str, optional
        Name of the new column indicating the split
    num_partitions : int, optional
        Sets the number of partitions to split with. Each partition needs
        to be some minimum size for averages to work out to an evenly split
        final set. (Default: 100)

    Returns
    -------
    pyspark.sql.DataFrame
        Input dataframe with split indices assigned to a new column
    """
    # General sanity check on provided splits. We could attempt
    # to normalize instead of fail, but this is good enough.
    assert abs(1 - sum(splits)) < 0.01

    mjolnir.spark.assert_columns(df, ['wikiid', 'norm_query_id'])

    def split_rows(rows):
        # Current number of items per split
        split_counts = defaultdict(lambda: [0] * len(splits))
        # starting at 1 prevents div by zero. Using a float allows later
        # division to work as expected.
        processed = defaultdict(lambda: 1.)
        for row in rows:
            # Assign row to first available split that has less than
            # the desired weight
            for i, percent in enumerate(splits):
                if split_counts[row.wikiid][i] / processed[row.wikiid] < percent:
                    split_counts[row.wikiid][i] += row.weight
                    yield (row.wikiid, row.norm_query_id, i)
                    break
            # If no split found assign to first split
            else:
                split_counts[row.wikiid][0] += row.weight
                yield (row.wikiid, row.norm_query_id, 0)
            processed[row.wikiid] += row.weight

    # Calculating splits with mapPartitions is only deterministic if the # of input
    # partitions stays the same but it seems catalyst can sometimes decide to change
    # the number of partitions of the input if it comes from disk based on the rest
    # of the plan.
    # This fights back by calculating everything on the driver. Maybe not ideal
    # but seems to work as we can guarantee the splits are calculated a single time.
    # This is reasonable because even with 10's of M of samples there will only
    # be a few hundred thousand (wikiid, norm_query_id) rows.
    rows = (
        df
        .groupBy('wikiid', 'norm_query_id')
        .agg(F.count(F.lit(1)).alias('weight'))
        .collect())

    df_splits = (
        df._sc.parallelize(split_rows(rows))
        .toDF(['wikiid', 'norm_query_id', output_column]))

    return df.join(df_splits, how='inner', on=['wikiid', 'norm_query_id'])


def group_k_fold(df, num_folds, num_partitions=100, output_column='fold'):
    """
    Generates group k-fold splits. The fold a row belongs to is
    assigned to the column identified by the output_column parameter.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    num_folds : int
    test_folds : int, optional
    vali_folds : int, optional
    num_partitions : int, optional

    Yields
    ------
    dict
    """
    return (
        split(df, [1. / num_folds] * num_folds, output_column, num_partitions)
        .withColumn(output_column, mjolnir.spark.add_meta(df._sc, F.col(output_column), {
            'num_folds': num_folds,
        })))


def _make_folds(df, num_folds, num_workers, pool):
    """Transform a DataFrame with assigned folds into many dataframes.

    The results of split and group_k_fold emit a single dataframe with folds
    marked on the individual rows. To do the resulting training we need individual
    dataframes for each test/train split within the folds. If the data has
    not already had folds assigned they will be assigned based on the 'num_folds'
    key in params. If not present 5 folds will be used.

    Also generates the appropriate group data for xgboost. This doesn't
    necessarily belong here, but it is relatively expensive to calculate, so we
    benefit significantly by doing it once before hyperparameter tuning, as
    opposed to doing it for each iteration.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input dataframe with a 'fold' column indicating which fold each row
        belongs to.
    num_folds : int
        Number of folds to create. If a 'fold' column already exists in df
        this will be ignored.
    num_workers : int
        Number of workers used to train each model. This is passed onto
        xgboost.prep_training to prepare each fold.
    pool : multiprocessing.dummy.Pool or None
        Used to prepare folds in parallel. If not provided folds will be
        generated sequentially.

    Returns
    -------
    list
        Generates a list of dicts, one for each fold. Each dict contains
        train and test keys containing DataFrames, along with j_train_groups
        and j_test_groups keys which contain py4j JavaObject instances
        corresponding to group data needed by xgboost for train/eval. The train
        and test dataframes have been persisted, so should be unpersisted
        when no longer needed.
    """
    if 'fold' in df.columns:
        assert num_folds == df.schema['fold'].metadata['num_folds']
        df_folds = df
    else:
        df_folds = group_k_fold(df, num_folds)

    def job(fold):
        condition = F.col('fold') == fold
        # TODO: de-couple xgboost from cv generation.
        df_train, j_train_groups = mjolnir.training.xgboost.prep_training(
                df_folds.where(~condition), num_workers)
        df_test, j_test_groups = mjolnir.training.xgboost.prep_training(
                df_folds.where(condition), num_workers)
        return {
            'train': df_train,
            'test': df_test,
            'j_train_groups': j_train_groups,
            'j_test_groups': j_test_groups
        }

    if pool is None:
        return map(job, range(num_folds))
    else:
        return pool.map(job, range(num_folds))


def _py4j_retry(fn, default_retval):
    """Wrap a function utilizing py4j with retry handling

    Race condition in py4j? some other problem? Not sure yet, only seems
    to be triggered with multiple python threads and higher rounds for
    xgboost. Is fairly rare, but with enough hyperopt iterations it is
    somewhat common
    """
    def with_retry(*args, **kwargs):
        failures = 0
        while failures < 2:
            try:
                return fn(*args, **kwargs)
            except py4j.protocol.Py4JJavaError as e:
                print e
                failures += 1
        return default_retval
    return with_retry


def _cross_validate(folds, train_func, params, num_workers, pool):
    """Perform cross validation of the provided folds

    Parameters
    ----------
    folds : list
    train_func : callable
    params : dict
    num_workers : int
    pool : multiprocessing.dummy.Pool or None

    Returns
    -------
    list
    """
    def job(fold):
        local_params = params.copy()
        local_params['groupData'] = fold['j_train_groups']
        model = train_func(fold['train'], local_params, num_workers=num_workers)
        return {
            'train': model.eval(fold['train'], fold['j_train_groups']),
            'test': model.eval(fold['test'], fold['j_test_groups']),
        }

    job_w_retry = _py4j_retry(job, {
        'train': float('nan'),
        'test': float('nan'),
    })

    if pool is None:
        return map(job_w_retry, folds)
    else:
        return pool.map(job_w_retry, folds)


def cross_validate(df, train_func, params, num_folds=5, num_workers=5, pool=None):
    """Perform cross-validation of the dataframe

    Parameters
    ----------
    df : pyspark.sql.DataFrame or list
    train_func : callable
        Function used to train a model. Must return a model that
        implements an eval method
    params : dict
        parameters to pass on to train_func
    num_folds : int
        Number of folds to split df into for cross validation
    num_workers : int
        Number of executors to use for each model training
    pool : multiprocessing.dummy.Pool, optional
        Used to prepare folds and run cross validations in parallel. If
        not provided all work will be done sequentially.

    Returns
    -------
    list
        List of dicts, each dict containing a train and test key. The values
        correspond the the model evaluation metric for the train and test
        data frames.
    """
    folds = _make_folds(df, num_folds, num_workers, pool)
    try:
        return _cross_validate(folds, train_func, params, num_workers=num_workers, pool=pool)
    finally:
        for fold in folds:
            fold['train'].unpersist()
            fold['test'].unpersist()
