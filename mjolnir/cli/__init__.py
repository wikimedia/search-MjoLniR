from argparse import ArgumentParser
from typing import cast, Callable, Dict

from . import kafka_bulk_daemon, kafka_msearch_daemon

Factory = Callable[[ArgumentParser], Callable[..., None]]
CLI_COMMANDS = cast(Dict[str, Factory], {
    'kafka_bulk_daemon': kafka_bulk_daemon.configure,
    'kafka_daemon': kafka_bulk_daemon.configure,
    'kafka_msearch_daemon': kafka_msearch_daemon.configure,
})

try:
    # None of these work if pyspark is unavailable, and we don't
    # want to force pyspark to be available for the kafka daemons
    from . import (
        dbn, feature_selection, feature_vectors, hyperparam,
        make_folds, norm_query_clustering, query_clicks_ltr, train
    )

    CLI_COMMANDS.update({
        'dbn': dbn.configure,
        'feature_selection': feature_selection.configure,
        'feature_vectors': feature_vectors.configure,
        'hyperparam': hyperparam.configure,
        'make_folds': make_folds.configure,
        'norm_query_clustering': norm_query_clustering.configure,
        'query_clicks_ltr': query_clicks_ltr.configure,
        'train': train.configure,
    })
except ImportError as e:
    if e.name is None or not e.name.startswith('pyspark'):
        raise
