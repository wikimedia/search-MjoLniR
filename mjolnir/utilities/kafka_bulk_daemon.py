"""
Daemon to collect elasticsearch bulk indexing requests from kafka
and push them into elasticsearch.
"""
import argparse
from collections import namedtuple
import logging
import re
import time

from elasticsearch import Elasticsearch
from mjolnir.kafka import bulk_daemon
import prometheus_client

log = logging.getLogger(__name__)
MemoizeEntry = namedtuple('MemoizeEntry', ('value', 'valid_until'))


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


def make_es_clusters(bootstrap_hosts):
    """Return all clusters reachable through bootstrap hosts

    Parameters
    ----------
    bootstrap_hosts : str
            CSV of host:port pairs to bootstrap from. Each must be for a different cluster.

    Returns
    -------
    List[Elasticsearch]

    """
    clusters = [Elasticsearch(host) for host in bootstrap_hosts.split(',')]
    seen = set()
    for cluster in clusters:
        info = cluster.info()
        if info['cluster_uuid'] in seen:
            raise ValueError(
                'Cluster %s (uuid %s) seen from more than one bootstrap host',
                info['cluster_name'], info['cluster_uuid'])
        seen.add(info['cluster_uuid'])
        log.info('Connected to elasticsearch %s', info['cluster_name'])
        # FIXME: Stop injecting properties on other peopls objects
        # when we stop supporting elastic 5.x
        cluster.major_version = int(info['version']['number'].split('.')[0])
    for cluster in clusters:
        new_hosts = get_hosts_from_crosscluster_conf(cluster.cluster.get_settings())
        for name, new_host in new_hosts.items():
            info = new_host.info()
            if info['cluster_uuid'] in seen:
                continue
            seen.add(info['cluster_uuid'])
            new_host.major_version = int(info['version']['number'].split('.')[0])
            clusters.append(new_host)
    return clusters


def to_http_url(host):
    if not re.match('^[a-z0-9.]+:9[3579]00$', host):
        raise ValueError("Invalid hostname {}".format(host))
    hostname, port = host.split(':', 1)
    port = int(port)
    # We don't have certs for elastic hostnames only LVS...
    return 'http://{}:{}'.format(hostname, port - 100)


def get_hosts_from_crosscluster_conf(conf):
    elastichosts = {}
    cross_clusters = {}

    try:
        cross_clusters = conf['persistent']['search']['remote']
    except KeyError:
        pass

    for cluster_name, cluster_conf in cross_clusters.items():
        hosts = [to_http_url(host) for host in cluster_conf['seeds']]
        elastichosts[cluster_name] = Elasticsearch(hosts)

    return elastichosts


def ttl_memoize(f, **kwargs):
    TTL = 300
    cache = {}

    def memoized(*args):
        now = time.time()
        if args in cache:
            entry = cache[args]
            if entry.valid_until > now:
                return entry.value
        value = f(*args, **kwargs)
        cache[args] = MemoizeEntry(value, now + TTL)
        return value

    return memoized


def indices_map(clusters):
    """Map from addressable index name to elasticsearch client that contains it

    Index names that exist on multiple clusters are treated as existing on
    no clusters. Essentially this only tracks indices that are unique to
    the cluster it lives on.
    """
    indices = {}
    for elastic in clusters:
        for index_name, data in elastic.indices.get_alias().items():
            for name in [index_name] + list(data['aliases'].keys()):
                if name not in indices:
                    indices[name] = elastic
                # If an index name exists on multiple clusters we
                # pretend it doesn't exist on any of them.
                elif indices[name] != elastic:
                    indices[name] = None
    return {k: v for k, v in indices.items() if v is not None}


def main(brokers, es_clusters, topics, group_id, prometheus_port):
    clusters = make_es_clusters(es_clusters)
    indices_map_memo = ttl_memoize(indices_map, clusters=clusters)

    def client_for_index(name: str) -> Elasticsearch:
        return indices_map_memo()[name]

    prometheus_client.start_http_server(prometheus_port)
    bulk_daemon.run(brokers, client_for_index, topics, group_id)


if __name__ == "__main__":
    logging.basicConfig()
    kwargs = dict(vars(arg_parser().parse_args()))
    main(**kwargs)
