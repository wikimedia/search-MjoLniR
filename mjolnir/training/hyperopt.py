"""
Integration and additional features for the hyperopt library
"""
from __future__ import absolute_import
import hyperopt
import hyperopt.pyll.base
from hyperopt.utils import coarse_utcnow
import math
import mjolnir.training.tuning
import numpy as np


# FMinIter, when used async, puts the domain into attachments. Unfortunately
# this domain isn't picklable in our use case. We don't actually need it
# to be picklable, but FMinIter pickles it anways. Hax around it by setting
# async after FMinIter.__init__ runs and providing domain manually.
if not hasattr(hyperopt.FMinIter, '_mjolnir_hack'):
    # Alternatively, could inherit from FMinIter, then replace?
    hyperopt.FMinIter._mjolnir_hack = True
    fminiter_orig_init = hyperopt.FMinIter.__init__

    def _new_fminiter_init(self, *args, **kwargs):
        fminiter_orig_init(self, *args, **kwargs)
        if type(self.trials) == ThreadingTrials:
            # We have to set this here, rather than letting it
            # autodetect from self.trials.async, because then
            # it will try, and fail, to pickle the domain object
            self.async = True
            # Since domain wasn't pickled and provided we have
            # to do it manually
            self.trials.attachments['domain'] = self.domain
    hyperopt.FMinIter.__init__ = _new_fminiter_init


class ThreadingTrials(hyperopt.Trials):
    def __init__(self, pool):
        super(ThreadingTrials, self).__init__()
        self.pool = pool

    def _evaluate_one(self, trial):
        if trial['state'] != hyperopt.JOB_STATE_NEW:
            return
        trial['state'] = hyperopt.JOB_STATE_RUNNING
        now = coarse_utcnow()
        trial['book_time'] = now
        trial['refresh_time'] = now
        spec = hyperopt.base.spec_from_misc(trial['misc'])
        ctrl = hyperopt.base.Ctrl(self, current_trial=trial)
        try:
            result = self.attachments['domain'].evaluate(spec, ctrl)
        except Exception as e:
            trial['state'] = hyperopt.JOB_STATE_ERROR
            trial['misc']['error'] = (str(type(e)), str(e))
            trial['misc']['e'] = e
            trial['refresh_time'] = coarse_utcnow()
        else:
            trial['state'] = hyperopt.JOB_STATE_DONE
            trial['result'] = result
            trial['refresh_time'] = coarse_utcnow()

    def _insert_trial_docs(self, docs):
        rval = super(ThreadingTrials, self)._insert_trial_docs(docs)
        self.pool.imap_unordered(self._evaluate_one, docs)
        self.refresh()
        return rval

    def __getstate__(self):
        # This will be called by pickle. Attempting to pickle the domain object
        # will error out, as the objective function isn't top-level. The pool
        # is also un-picklable. Clear out these pieces to allow pickling to work.
        # Note that this object will basically be unusable for minimization after
        # unpickling, it will only be a data carrier at that point.
        state = self.__dict__.copy()
        del state['pool']
        state['attachments'] = state['attachments'].copy()
        try:
            del state['attachments']['domain']
        except KeyError:
            pass
        return state


def minimize(df, train_func, space, max_evals=50, algo=hyperopt.tpe.suggest,
             num_folds=5, num_workers=5, cv_pool=None, trials_pool=None):
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
    num_workers : int
        Number of executors to use for each model training
    cv_pool : multiprocessing.dummy.Pool or None
        Controls the number of models to run in parallel. If None models
        are trained sequentially.
    trials_pool : multiprocessing.dummy.Pool or None
        Controls the number of hyperopt trials run in parallel. If None trials
        are run sequentially.

    Returns
    -------
    best_params : dict
        The best parameters found within space
    trials : hyperopt.Trials
        Information about every iteration of the search
    """

    def objective(params):
        scores = mjolnir.training.tuning._cross_validate(
            folds, train_func, params, num_workers=num_workers,
            pool=cv_pool)
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
        df, num_folds=num_folds, num_workers=num_workers, pool=cv_pool)

    try:
        if trials_pool is None:
            trials = hyperopt.Trials()
        else:
            trials = ThreadingTrials(trials_pool)
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
