"""
Daemon to collect elasticsearch bulk indexing requests from kafka
and push them into elasticsearch.
"""
import argparse
import logging

import mjolnir.kafka.bulk_daemon


def arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-b', '--brokers', dest='brokers', required=True, type=str,
        help='Kafka brokers to bootstrap from as a comma separated list of <host>:<port>')
    parser.add_argument(
       '-c', '--es-clusters', dest='es_clusters', required=True, type=str,
       help='Elasticsearch servers to bootstrap from as a comma separated list of <host>:<port>')
    parser.add_argument(
        '-t', '--topic', dest='topics', required=True, type=str, nargs='+',
        help='Kafka topic(s) to read indexing requests from. Multiple topics may be provided.')
    parser.add_argument(
        '-g', '--group-id', dest='group_id', type=str, default='TODO',
        help='Kafka consumer group to join')
    parser.add_argument(
        '--prometheus-port', dest='prometheus_port', default=9170, type=int, required=False,
        help='Port to export prometheus metrics over.')
    return parser


main = mjolnir.kafka.bulk_daemon.run


if __name__ == "__main__":
    logging.basicConfig()
    kwargs = dict(vars(arg_parser().parse_args()))
    main(**kwargs)
