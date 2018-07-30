"""BC alias for previous msearch_daemon command name"""
import logging
from mjolnir.utilities.kafka_msearch_daemon import arg_parser, main

if __name__ == '__main__':
    logging.basicConfig()
    kwargs = dict(vars(arg_parser().parse_args()))
    main(**kwargs)
