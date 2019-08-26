import inspect

from mjolnir.utilities.kafka_msearch_daemon import arg_parser
from mjolnir.kafka.msearch_daemon import Daemon


def test_args_match_init():
    kwargs = dict(vars(arg_parser().parse_args(['-b', 'localhost:9092'])))
    available = set(inspect.getfullargspec(Daemon.__init__).args[1:])
    assert all(k in available for k in kwargs.keys())
