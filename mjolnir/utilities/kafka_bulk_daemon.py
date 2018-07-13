"""
Daemon to collect elasticsearch bulk indexing requests from kafka
and push them into elasticsearch.
"""
from __future__ import absolute_import
import argparse
import mjolnir.kafka.bulk_daemon


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-b', '--brokers', dest='brokers', required=True, type=str,
        help='Kafka brokers to bootstrap from as a comma separated list of <host>:<port>')
    parser.add_argument(
       '-c', '--es-clusters', dest='es_clusters', required=True, type=str,
       help='Elasticsearch servers to bootstrap from as a comma separated list of <host>:<port>')
    parser.add_argument(
        '-t', '--topic', dest='topic', required=True, type=str,
        help='Kafka topic to read indexing requests from')
    parser.add_argument(
        '-g', '--group-id', dest='group_id', type=str, default='TODO',
        help='Kafka consumer group to join')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    mjolnir.kafka.bulk_daemon.run(**dict(vars(args)))


if __name__ == "__main__":
    main()
