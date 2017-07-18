"""
Integration and additional features for the hyperopt library
"""
from __future__ import absolute_import
import hyperopt
import hyperopt.pyll.base
import itertools
import math
import mjolnir.training.tuning
import numpy as np


class _GridSearchAlgo(object):
    def __init__(self, space):
        foo = {}
        for k, v in space.items():
            if not isinstance(v, hyperopt.pyll.base.Apply):
                continue
            literals = v.pos_args[1:]
            if not all([isinstance(l, hyperopt.pyll.base.Literal) for l in literals]):
                raise ValueError('GridSearch only works with hp.choice')
            foo[k] = range(len(literals))
        self.grid_keys = foo.keys()
        self.grids = list(itertools.product(*foo.values()))
        self.max_evals = len(self.grids)

    def __call__(self, new_ids, domain, trials, seed):
        rval = []
        for ii, new_id in enumerate(new_ids):
            vals = dict(zip(self.grid_keys, [[v] for v in self.grids.pop()]))
            new_result = domain.new_result()
            new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir,
                            idxs=dict(zip(self.grid_keys, [[new_id]] * len(vals))),
                            vals=vals)
            rval.extend(trials.new_trial_docs([new_id],
                        [None], [new_result], [new_misc]))
        return rval


def grid_search(df, train_func, space, num_folds=5, num_fold_partitions=100,
                num_cv_jobs=5, num_workers=5):
    # TODO: While this tried to look simple, hyperopt is a bit odd to integrate
    # with this directly. Perhaps implement naive gridsearch directly instead
    # of through hyperopt.
    algo = _GridSearchAlgo(space)
    return minimize(df, train_func, space, algo.max_evals, algo, num_folds,
                    num_fold_partitions, num_cv_jobs, num_workers)


def minimize(df, train_func, space, max_evals=50, algo=hyperopt.tpe.suggest,
             num_folds=5, num_fold_partitions=100, num_cv_jobs=5, num_workers=5):
    """Perform cross validated hyperparameter optimization of train_func

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Features and Labels to optimize over
    train_func : callable
        Function to use for training individual models
    space : dict
        Hyperparameter space to search over.
    max_evals : int
        Maximum iterations of hyperparameter tuning to perform.
    algo : callable
        The algorithm to use with hyperopt. See docs of hyperopt.fmin for more
        details.
    num_folds : int
        Number of folds to split df into for cross validation
    num_fold_partitions : int
        Sets the number of partitions to split with. Each partition needs
        to be some minimum size for averages to work out to an evenly split
        final set. (Default: 100)
    num_cv_jobs : int
        Number of cross validation folds to train in parallel
    num_workers : int
        Number of executors to use for each model training
    cache : bool
        True if the folds of df should be individually cached
    unpersist : bool
        True if the folds of df should be unpersisted when complete.

    Returns
    -------
    best_params : dict
        The best parameters found within space
    trials : hyperopt.Trials
        Information about every iteration of the search
    """
    def objective(params):
        scores = mjolnir.training.tuning._cross_validate(
            folds, train_func, params, num_cv_jobs=num_cv_jobs,
            num_workers=num_workers)
        # For now the only metric is NDCG, and hyperopt is a minimizer
        # so return the negative NDCG
        loss = [-s['test'] for s in scores]
        true_loss = [s['train'] - s['test'] for s in scores]
        num_failures = sum([math.isnan(s) for s in loss])
        if num_failures > 1:
            return {
                'status': hyperopt.STATUS_FAIL,
                'failure': 'Too many failures: %d' % (num_failures)
            }
        else:
            loss = [s for s in loss if not math.isnan(s)]
            true_loss = [s for s in true_loss if not math.isnan(s)]
        return {
            'status': hyperopt.STATUS_OK,
            'loss': sum(loss) / float(len(loss)),
            'loss_variance': np.var(loss),
            'true_loss': sum(true_loss) / float(len(true_loss)),
            'true_loss_variance': np.var(true_loss),
        }

    folds = mjolnir.training.tuning._make_folds(
        df, num_folds=num_folds, num_workers=num_workers,
        num_fold_partitions=num_fold_partitions, num_cv_jobs=num_cv_jobs)

    for fold in folds:
        fold['train'].cache()
        fold['test'].cache()

    try:
        trials = hyperopt.Trials()
        best = hyperopt.fmin(objective, space, algo=algo,
                             max_evals=max_evals, trials=trials)
    finally:
        for fold in folds:
            fold['train'].unpersist()
            fold['test'].unpersist()

    # hyperopt only returns the non-constant parameters in best. It seems
    # more convenient to return all of them.
    best_merged = space.copy()
    best_merged.update(best)

    return (best_merged, trials)
