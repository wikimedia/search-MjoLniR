from argparse import ArgumentParser
from typing import cast, Callable, Mapping

from . import (
    dbn, feature_selection, feature_vectors, hyperparam,
    kafka_bulk_daemon, kafka_msearch_daemon, make_folds,
    norm_query_clustering, query_clicks_ltr, train
)

Factory = Callable[[ArgumentParser], Callable[..., None]]
CLI_COMMANDS = cast(Mapping[str, Factory], {
    'dbn': dbn.configure,
    'feature_selection': feature_selection.configure,
    'feature_vectors': feature_vectors.configure,
    'hyperparam': hyperparam.configure,
    'kafka_bulk_daemon': kafka_bulk_daemon.configure,
    'kafka_daemon': kafka_bulk_daemon.configure,
    'kafka_msearch_daemon': kafka_msearch_daemon.configure,
    'make_folds': make_folds.configure,
    'norm_query_clustering': norm_query_clustering.configure,
    'query_clicks_ltr': query_clicks_ltr.configure,
    'train': train.configure,
})
