"""
Provides access to a set of utilities for running mjolnir pipelines.

Utilities:

* spark                Wrapper around spark-submit and a configuration file
*                      to run commands in expected environments.
* upload               Upload trained models to elasticsearch
* data_pipeline        Individual spark job for converting click data into
*                      labeled training data.
* training_pipeline    Individual spark job for turning labeled training data
*                      into xgboost models.
* kafka_msearch_daemon Daemon side of feature collection via kafka
* kafka_bulk_daemon    Daemon for pushing document updates from kafka into
*                      elasticsearch.

Usage:
    mjolnir (-h | --help)
    mjolnir <utility> [-h|--help]
"""  # noqa

from __future__ import absolute_import
import logging
import logging.config
import os
import sys
import traceback
import pyyaml
from importlib import import_module

USAGE = """Usage:
    mjolnir (-h | --help)
    mjolnir <utility> [-h|--help]\n"""


DEFAULT_LOGGING_CONFIG = [os.path.join(d, "logging_config.yaml") for d in ('./', '/etc/mjolnir')]
DEFAULT_LOGGING_FORMAT = "%(asctime)s %(levelname)s:%(name)s -- %(message)s"

log = logging.getLogger(__name__)


def configure_logging(log_level=None, logging_config=None, **kwargs):
    if logging_config is None:
        for path in DEFAULT_LOGGING_CONFIG:
            if os.path.exists(path):
                logging_config = path
                break

    if logging_config is None:
        logging.basicConfig(level=logging.INFO, format=DEFAULT_LOGGING_FORMAT)
        # requests is spammy past info
        logging.getLogger('requests').setLevel(logging.INFO)
    else:
        log.info('Loading logging configuration from %s', logging_config)
        with open(logging_config) as f:
            logging.config.dictConfig(pyyaml.load(f))

        # If running from console mirror logs there
        if sys.stdin.isatty():
            handler = logging.StreamHandler(stream=sys.stderr)
            formatter = logging.Formatter(fmt=DEFAULT_LOGGING_FORMAT)
            handler.setFormatter(formatter)
            logging.getLogger().addHandler(handler)

    if log_level:
        logging.getLogger().setLevel(log_level)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if not len(argv):
        sys.stderr.write(USAGE)
        sys.exit(1)
    elif argv[0] in ("-h", "--help"):
        sys.stderr.write(__doc__ + "\n")
        sys.exit(1)
    elif argv[0][:1] == "-":
        sys.stderr.write(USAGE)
        sys.exit(1)

    module_name = argv.pop(0)
    try:
        module = import_module(".utilities." + module_name,
                               package="mjolnir")
    except ImportError:
        sys.stderr.write(traceback.format_exc())
        sys.stderr.write("Could not find utility %s" % (module_name))
        sys.exit(1)

    parser = module.arg_parser()
    parser.add_argument(
        '-v', '--verbose', dest='log_level', action='store_const',
        const='DEBUG', help='Increase logging level to DEBUG')
    parser.add_argument(
        '--logging-conf', dest='logging_config',
        help='Path to logging configuration.')
    kwargs = dict(vars(parser.parse_args(argv)))
    configure_logging(**kwargs)
    del kwargs['log_level']
    del kwargs['logging_config']
    module.main(**kwargs)


if __name__ == "__main__":
    main()
