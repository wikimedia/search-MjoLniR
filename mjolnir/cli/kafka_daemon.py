"""
Daemon to collect elasticsearch bulk requests from kafka,
run them against relforge, and send the results back over
kafka.
"""

import argparse
import logging
import mjolnir.kafka.daemon


def parse_arguments():
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument(
        '-b', '--brokers', dest='brokers', required=True, type=lambda x: x.split(','),
        help='Kafka brokers to bootstrap from as a comma separated list of <host>:<port>')
    parser.add_argument(
        '-m', '--max-request-size', dest='max_request_size', type=int, default=4*1024*1024*10,
        help='Max size of requets sent to the kafka broker'
             + 'Defaults to 40MB.')
    parser.add_argument(
        '-w', '--num-workers', dest='n_workers', type=int, default=5,
        help='Number of workers to issue elasticsearch queries in parallel. '
             + 'Defaults to 5.')
    parser.add_argument(
        '-v', '--verbose', dest='verbose', default=False, action='store_true',
        help='Increase logging to INFO')
    parser.add_argument(
        '-vv', '--very-verbose', dest='very_verbose', default=False, action='store_true',
        help='Increase logging to DEBUG')
    args = parser.parse_args()
    return dict(vars(args))


if __name__ == '__main__':
    args = parse_arguments()
    if args['very_verbose']:
        logging.basicConfig(level=logging.DEBUG)
    elif args['verbose']:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig()
    del args['verbose']
    del args['very_verbose']
    mjolnir.kafka.daemon.Daemon(**args).run()
