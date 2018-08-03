from collections import namedtuple
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
import json
import jsonschema
import kafka
import logging
import mjolnir.kafka
import time


log = logging.getLogger(__name__)
MemoizeEntry = namedtuple('MemoizeEntry', ('value', 'valid_until'))

# Fields we accept updates for, found in the _source field of incoming
# messages, and their configuration for the noop plugin.
FIELD_CONFIG = {
    'popularity_score': 'within 20%',
}

# jsonschema of incoming requests
VALIDATOR = jsonschema.Draft4Validator({
    "type": "object",
    "additionalProperties": False,
    "required": ["_index", "_id", "_source"],
    "properties": {
        "_index": {"type": "string"},
        "_id": {"type": ["integer", "string"]},
        "_source": {
            "type": "object",
            "additionalProperties": False,
            "minProperties": 1,
            "properties": {field: {"type": ["number", "string"]} for field in FIELD_CONFIG.keys()}
        }
    }
})


def expand_action(message):
    """Transform an update request into an es bulk update"""
    action = {
        'update': {
            '_index': message['_index'],
            '_type': 'page',
            '_id': message['_id'],
        }
    }

    noop_handlers = {field: FIELD_CONFIG[field] for field in message['_source'].keys()}
    source = {
        'script': {
            'inline': 'super_detect_noop',
            'lang': 'native',
            'params': {
                'handlers': noop_handlers,
                'source': message['_source'],
            }
        }
    }

    return action, source


def stream_to_es(cluster, records):
    # This will throw exceptions for any error connecting to
    # elasticsearch (perhaps a rolling restart?). In that case the
    # daemon will shut down and be restarted by systemd. Rebalancing
    # will assign the partition to another daemon that hopefully isn't
    # having connection issues.
    for ok, result in streaming_bulk(
        client=cluster,
        actions=records,
        raise_on_error=False,
        expand_action_callback=expand_action,
    ):
        action, result = result.popitem()
        status = result.get('status', 500)
        # 404 are quite common unfortunately. The analytics side doesn't
        # know the namespace mappings and attempts to send all updates
        # to <wiki>_content, letting the docs that don't exist fail.
        if not ok and status != 404:
            log.warning('Failed elasticsearch %s request: %s', action, str(result)[:512])


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


def available_indices(cluster):
    """Returns the set of addressable indices and aliases."""
    indices = set()
    for index_name, data in cluster.indices.get_alias().items():
        indices.add(index_name)
        indices.update(data['aliases'].keys())
    return indices


def split_records_by_cluster(indices_on_clusters, poll_response):
    """Split a poll response from kafka into per-es-cluster batches"""
    split = [[] for _ in indices_on_clusters]
    for records in poll_response.values():
        for record in records:
            try:
                value = json.loads(record.value.decode('utf-8'))
            except ValueError:
                log.warning('Invalid message: %s', record.value[:128])
                continue

            errors = list(VALIDATOR.iter_errors(value))
            if errors:
                log.warning('\n'.join(map(str, errors)))
                continue

            for i, indices in enumerate(indices_on_clusters):
                if value['_index'] in indices:
                    split[i].append(value)
                    break
            else:
                log.warning('Could not find cluster for index %s', value['_index'])
    return split


def make_es_clusters(bootstrap_hosts):
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
    return clusters


def run(brokers, es_clusters, topics, group_id):
    es_clusters = make_es_clusters(es_clusters)
    all_available_indices_memo = ttl_memoize(lambda: [available_indices(c) for c in es_clusters])
    consumer = kafka.KafkaConsumer(
        bootstrap_servers=brokers,
        group_id=group_id,
        # Commits are manually performed for each batch returned by poll()
        # after they have been processed by elasticsearch.
        enable_auto_commit=False,
        # If we lose the offset safest thing is to replay from
        # the beginning. In WMF this is typically 7 days, the
        # same lifetime as offsets.
        auto_offset_reset='earliest',
        api_version=mjolnir.kafka.BROKER_VERSION,
        # Our expected records are tiny and compress well. Accept
        # large batches. Increased from default of 500.
        max_poll_records=2000,
    )

    log.info('Subscribing to: %s', ', '.join(topics))
    consumer.subscribe(topics)
    try:
        while True:
            poll_response = consumer.poll(timeout_ms=60000)
            # Did the poll time out?
            if not poll_response:
                continue
            # Possibly refresh our information about what indices live where
            all_available_indices = all_available_indices_memo()
            # Figure out what cluster everything goes to
            split = split_records_by_cluster(
                all_available_indices, poll_response)
            # Send to the cluster
            for cluster, records in zip(es_clusters, split):
                if records:
                    stream_to_es(cluster, records)
            # Tell kafka we did the work
            offsets = {}
            for tp, records in poll_response.items():
                offsets[tp] = kafka.OffsetAndMetadata(records[-1].offset + 1, '')
            consumer.commit_async(offsets)
    finally:
        consumer.close()
