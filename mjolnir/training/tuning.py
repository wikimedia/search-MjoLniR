"""
Support for making test/train or k-fold splits
"""
from __future__ import absolute_import
from collections import defaultdict
import mjolnir.spark
import py4j.protocol
from pyspark.sql import functions as F


def split(df, splits, output_column='fold'):
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


def group_k_fold(df, num_folds, output_column='fold'):
    """
    Generates group k-fold splits. The fold a row belongs to is
    assigned to the column identified by the output_column parameter.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    num_folds : int
    output_column : str, optional

    Yields
    ------
    dict
    """
    return (
        split(df, [1. / num_folds] * num_folds, output_column)
        .withColumn(output_column, mjolnir.spark.add_meta(df._sc, F.col(output_column), {
            'num_folds': num_folds,
        })))


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


def cross_validate(folds, train_func, params, pool):
    """Perform cross validation of the provided folds

    Parameters
    ----------
    folds : list of dict containing train and test keys
    train_func : callable
    params : dict
    pool : multiprocessing.dummy.Pool or None

    Returns
    -------
    list
    """
    def job(fold):
        model = train_func(fold, params)
        # TODO: Summary is hardcodeed to train/test
        return {
            "train": model.summary().train(),
            "test": model.summary().test(),
        }

    job_w_retry = _py4j_retry(job, {
        "train": [float('nan')],
        "test": [float('nan')],
    })

    if pool is None:
        return map(job_w_retry, folds)
    else:
        return pool.map(job_w_retry, folds)
