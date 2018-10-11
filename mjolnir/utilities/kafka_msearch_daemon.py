"""
Daemon to collect elasticsearch bulk requests from kafka,
run them against relforge, and send the results back over
kafka.
"""

from __future__ import absolute_import
import argparse
import logging
import mjolnir.kafka.msearch_daemon


def arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-b', '--brokers', dest='brokers', required=True, type=lambda x: x.split(','),
        help='Kafka brokers to bootstrap from as a comma separated list of <host>:<port>')
    parser.add_argument(
        '-m', '--max-request-size', dest='max_request_size', type=int, default=4*1024*1024*10,
        help='Max size of requsets sent to the kafka broker'
             + 'Defaults to 40MB.')
    parser.add_argument(
        '-w', '--num-workers', dest='n_workers', type=int, default=1,
        help='Number of workers to issue elasticsearch queries in parallel. '
             + 'Defaults to 1.')
    parser.add_argument(
        '--prometheus-port', dest='prometheus_port', default=9171, type=int, required=False,
        help='Port to export prometheus metrics over.')
    parser.add_argument(
        '--input-topic', dest='topic_work', default=mjolnir.kafka.TOPIC_REQUEST,
        help='Topic to read msearch requests from')
    parser.add_argument(
        '--output-topic', dest='topic_result', default=mjolnir.kafka.TOPIC_RESULT,
        help='Topic to write msearch responses to')
    parser.add_argument(
        '--group-id', dest='group_id', default='mjolnir_msearch',
        help='Kafka consumer group to join')
    parser.add_argument(
        '--max-concurrent-searches', dest='max_concurrent_searches', type=int, default=1,
        help='Maximum number of queries in a single msearch request that will run in parallel')
    return parser


def main(**kwargs):
    mjolnir.kafka.msearch_daemon.Daemon(**kwargs).run()


if __name__ == '__main__':
    logging.basicConfig()
    kwargs = dict(vars(arg_parser().parse_args()))
    main(**kwargs)
