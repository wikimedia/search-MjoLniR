from argparse import ArgumentParser
from typing import cast, Callable, Mapping

from . import kafka_bulk_daemon, kafka_msearch_daemon

Factory = Callable[[ArgumentParser], Callable]
CLI_COMMANDS = cast(Mapping[str, Factory], {
    'kafka_bulk_daemon': kafka_bulk_daemon.configure,
    'kafka_daemon': kafka_bulk_daemon.configure,
    'kafka_msearch_daemon': kafka_msearch_daemon.configure,
})
