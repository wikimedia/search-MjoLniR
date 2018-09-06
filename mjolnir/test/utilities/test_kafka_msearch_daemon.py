import inspect

from mjolnir.utilities.kafka_msearch_daemon import arg_parser
import mjolnir.kafka.msearch_daemon

def test_args_match_init():
    kwargs = dict(vars(arg_parser().parse_args(['-b', 'localhost:9092'])))
    available = set(inspect.getargspec(mjolnir.kafka.msearch_daemon.Daemon.__init__).args[1:])
    assert all(k in available for k in kwargs.keys())


