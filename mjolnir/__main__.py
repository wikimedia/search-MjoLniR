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
import sys
import traceback
from importlib import import_module

USAGE = """Usage:
    mjolnir (-h | --help)
    mjolnir <utility> [-h|--help]\n"""


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    if not len(args):
        sys.stderr.write(USAGE)
        sys.exit(1)
    elif args[0] in ("-h", "--help"):
        sys.stderr.write(__doc__ + "\n")
        sys.exit(1)
    elif args[0][:1] == "-":
        sys.stderr.write(USAGE)
        sys.exit(1)

    module_name = args.pop(0)
    try:
        module = import_module(".utilities." + module_name,
                               package="mjolnir")
    except ImportError:
        sys.stderr.write(traceback.format_exc())
        sys.stderr.write("Could not find utility %s" % (module_name))
        sys.exit(1)

    logging.basicConfig(level='INFO')
    module.main(args)


if __name__ == "__main__":
    main()
