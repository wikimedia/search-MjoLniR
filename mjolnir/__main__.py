"""Provides access to a set of utilities for running mjolnir pipelines."""

from argparse import ArgumentParser
import logging
import logging.config
import signal
import sys
import traceback

from mjolnir.config import load_config
import mjolnir.cli

DEFAULT_LOGGING_FORMAT = "%(asctime)s %(levelname)s:%(name)s -- %(message)s"
log = logging.getLogger(__name__)


def configure_logging(log_level=None, logging_config=None, **kwargs):
    logging_config = load_config(logging_config, 'logging_config.yaml')

    if logging_config is None:
        logging.basicConfig(level=logging.INFO, format=DEFAULT_LOGGING_FORMAT)
        # requests is spammy past info
        logging.getLogger('requests').setLevel(logging.INFO)
    else:
        logging.config.dictConfig(logging_config)

        # If running from console mirror logs there
        if sys.stdin.isatty():
            handler = logging.StreamHandler(stream=sys.stderr)
            formatter = logging.Formatter(fmt=DEFAULT_LOGGING_FORMAT)
            handler.setFormatter(formatter)
            logging.getLogger().addHandler(handler)

    if log_level:
        logging.getLogger().setLevel(log_level)


def log_all_threads_stack_trace(signal_num, active_frame):
    # active_frame is also reported by _current_frames(), and identified by
    # having this function at the end of it's stack trace. As such we ignore
    # the active_frame arg
    for thread_id, frame in sys._current_frames().items():
        log.warning(
            'Stack for thread %d\n%s', thread_id,
            '\n'.join(traceback.format_stack(frame)))


def main(argv=None) -> int:
    parser = ArgumentParser()
    parser.add_argument(
        '-v', '--verbose', dest='log_level', action='store_const',
        const='DEBUG', help='Increase logging level to DEBUG')
    parser.add_argument(
        '--logging-conf', dest='logging_config',
        help='Path to logging configuration.')

    # Create a sub parser for each invokable command
    subparsers = parser.add_subparsers(dest='command')
    functions = {}
    for name, factory in mjolnir.cli.CLI_COMMANDS.items():
        functions[name] = factory(subparsers.add_parser(name))

    kwargs = dict(vars(parser.parse_args(argv)))
    configure_logging(**kwargs)
    if kwargs['command'] is None:
        parser.print_usage()
        return 1

    command = functions[kwargs['command']]

    del kwargs['command']
    del kwargs['log_level']
    del kwargs['logging_config']
    # Install a signal handler to help figure out whats going on
    # when a daemon gets stuck.
    signal.signal(signal.SIGUSR1, log_all_threads_stack_trace)

    command(**kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
