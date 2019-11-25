from argparse import ArgumentParser
import inspect

from mjolnir.cli.kafka_msearch_daemon import configure
from mjolnir.kafka.msearch_daemon import Daemon


def test_args_match_init():
    parser = ArgumentParser()
    configure(parser)
    kwargs = dict(vars(parser.parse_args(['-b', 'localhost:9092'])))

    available = set(inspect.getfullargspec(Daemon).args[1:])
    assert all(k in available for k in kwargs.keys())
