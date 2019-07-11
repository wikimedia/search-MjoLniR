"""Perform actions based on new file(s) becoming available in swift

Recieves notifications over kafka about new files that are available in swift
and performs actions with them. Actions might import those files into elasticsearch
indices.
"""

from collections import namedtuple
import gzip
import json
import logging
import os
import re
from tempfile import TemporaryDirectory

from elasticsearch.helpers import parallel_bulk
import elasticsearch.exceptions
import jsonschema
import kafka
import prometheus_client
from swiftclient.service import SwiftError, SwiftService

import mjolnir.kafka

log = logging.getLogger(__name__)
Message = namedtuple('Message', ('container', 'object_prefix', 'action'))

CONFIG = {
    # key is the name of the swift container files were uploaded to
    'search_popularity_score': lambda *args: ImportExistingIndices(*args).run(),
    'search_glent': lambda *args: ImportAndPromote(
        *args,
        # TODO: Should be externally configurable
        index_pattern='glent_{prefix}',
        alias='glent_production',
        alias_rollback='glent_rollback').run(),
}


class Metric:
    _INVALID_RECORDS = prometheus_client.Counter(
        'mjolnir_swift_invalid_records_total',
        "Number of requests that could not be processed",
        ['reason']
    )
    FAIL_VALIDATE = _INVALID_RECORDS.labels(reason="validate")
    FAIL_NO_CONFIG = _INVALID_RECORDS.labels(reason="no_config")

    PROCESS_MESSAGE = prometheus_client.Summary(
        'mjolnir_swift_process_message_seconds',
        'Time taken to process individual kafka messages')

    BULK_IMPORT = prometheus_client.Summary(
        'mjolnir_swift_import_file_seconds',
        'Time taken to import a file into elasticsearch'
    )

    _BULK_ACTION_RESULT = prometheus_client.Counter(
        'mjolnir_swift_action_total',
        'Number of bulk action responses per result type', ['result'])
    ACTION_RESULTS = {
        'updated': _BULK_ACTION_RESULT.labels(result='updated'),
        'created': _BULK_ACTION_RESULT.labels(result='created'),
        'noop': _BULK_ACTION_RESULT.labels(result='noop'),
    }
    OK_UNKNOWN = _BULK_ACTION_RESULT.labels(result='ok_unknown')
    MISSING = _BULK_ACTION_RESULT.labels(result='missing')
    FAILED = _BULK_ACTION_RESULT.labels(result='failed')


# namedtuple and jsonschema of incoming requests
VALIDATOR = jsonschema.Draft4Validator({
    "type": "object",
    "additionalProperties": False,
    "required": ["container", "object_prefix"],
    "properties": {
        "container": {"type": "string"},
        "object_prefix": {"type": "string"}
    }
})

SANITIZE_PATTERN = re.compile('[^a-zA-Z0-9]')


def sanitize_index_name(name):
    """Limit the characters allowed into index names for our own sanity"""
    return SANITIZE_PATTERN.sub('_', name)


def load_and_validate(poll_response, config=CONFIG):
    """Parse raw kafka records into valid Message instances"""
    for records in poll_response.values():
        for record in records:
            try:
                value = json.loads(record.value.decode('utf-8'))
            except UnicodeDecodeError:
                Metric.FAIL_VALIDATE.inc()
                log.exception("Invalid unicode in message: %s", str(record.value)[:1024])
                continue
            except ValueError:
                Metric.FAIL_VALIDATE.inc()
                log.warning('Invalid json in message: %s', record.value.decode('utf-8')[:1024])
                continue

            errors = list(VALIDATOR.iter_errors(value))
            if errors:
                Metric.FAIL_VALIDATE.inc()
                log.warning('\n'.join(map(str, errors)))
                continue

            try:
                action = config[value['container']]
            except KeyError:
                Metric.FAIL_NO_CONFIG.inc()
                log.warning("Unknown swift container: %s", value['container'])
                continue

            yield Message(value['container'], value['object_prefix'], action)


def download_from_swift(swift, message):
    """Download files from swift and yield them as they become available"""
    with TemporaryDirectory() as tempdir:
        try:
            for download in swift.download(container=message.container, options={
                'prefix': message.object_prefix,
                'out_directory': tempdir,
            }):
                if download.get('success'):
                    yield download['path']
                    os.unlink(download['path'])
                else:
                    log.critical('Failure downloading from swift: %s', str(download)[:1024])
                    raise Exception(download.get('error', 'Malformed response from swift'))
        except SwiftError:
            # TODO: Are some errors handleable?
            raise


def open_by_suffix(filename, mode):
    """Open a file for reading, taking into account suffixes such as .gz"""
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    else:
        return open(filename, mode)


def pair(it):
    """Yield pairs of values from the iterable

    Example: pair(range(4)) == ((0,1), (2,3))
    """
    it = iter(it)
    return zip(it, it)


@Metric.BULK_IMPORT.time()
def bulk_import(**kwargs):
    """Bulk import data to elasticsearch.

    Tracks bulk import response metrics, reporting both externally to
    prometheus and to the caller.
    """
    log.info('Starting bulk import')
    good, missing, errors = 0, 0, 0
    for ok, result in parallel_bulk(raise_on_error=False, **kwargs):
        action, result = result.popitem()
        if ok:
            good += 1
            try:
                Metric.ACTION_RESULTS[result['result']].inc()
            except KeyError:
                Metric.OK_UNKNOWN.inc()
        elif result.get('status') == 404:
            # 404 are quite common so we log them separately. The analytics
            # side doesn't know the namespace mappings and attempts to send all
            # updates to <wiki>_content, letting the docs that don't exist fail
            missing += 1
            Metric.MISSING.inc()
        else:
            Metric.FAILED.inc()
            log.warning('Failed bulk %s request: %s', action, str(result)[:1024])
            errors += 1
    log.info('Completed import with %d success %d missing and %d errors', good, missing, errors)
    return good, missing, errors


class MalformedUploadException(Exception):
    """A file downloaded from swift cannot be processed"""
    pass


class ImportFailedException(Exception):
    """The import has failed and must not be retried"""
    pass


class UploadAction:
    """Perform an action based on a new file upload(s) becoming available"""
    def __init__(self, client_for_index, swift, message):
        self.client_for_index = client_for_index
        self.swift = swift
        self.message = message

    @Metric.PROCESS_MESSAGE.time()
    def run(self):
        self.pre_check()
        for path in download_from_swift(self.swift, self.message):
            self.on_file_available(path)
        self.on_download_complete()

    def pre_check(self):
        pass

    def on_file_available(self, path):
        raise NotImplementedError()

    def on_download_complete(self):
        pass


class ImportExistingIndices(UploadAction):
    """
    Import file(s) to existing indices spread across multiple clusters.
    Imported file(s) must specify both index name and doc type. A single
    file must not contain updates for multiple indices.
    """
    def on_file_available(self, path):
        try:
            with open_by_suffix(path, 'rt') as f:
                header = json.loads(next(iter(f)))
            action, meta = header.popitem()
            index_name = meta['_index']
        except (ValueError, KeyError):
            raise MalformedUploadException(
                "Loaded file is malformed and cannot be processed: {}".format(path))

        with open_by_suffix(path, 'rt') as f:
            # Ignoring errors, can't do anything useful with them. They still
            # get logged and counted.
            bulk_import(client=self.client_for_index(index_name),
                        actions=pair(line.strip() for line in f),
                        expand_action_callback=lambda x: x)


class ImportAndPromote(UploadAction):
    """
    Import file(s) to an elasticsearch index, and promote that index to own a specific alias.
    Imported file(s) must not specify index name or doc type.

    Process is roughly as follows:

    1. A kafka message is received indicating the swift container and prefix of files that must be imported.
    2. Referenced files are all pushed through the bulk indexing pipeline, expecting index auto creation
       with index templates to take care of configuring the index.
    3. The new index is promoted to production usage by changing an alias.
    4. Indices previously assigned to the prod alias are assigned to the rollback alias.
    5. Indices previously assigned to the rollback alias are deleted.

    While this tracks indices for rollback purposes, no rollbacking mechanism is provided.
    To perform a rollback an operator must use the elasticsearch index aliases api to clear
    the current production alias and add the indices from the rollback alias to the production
    alias. In this way the rollback alias i never used directly, but is instead used to hold
    the state of which index should be rolled back to.
    """

    def __init__(self, client_for_index, swift, message, index_pattern, alias, alias_rollback):
        super(ImportAndPromote, self).__init__(client_for_index, swift, message)
        self.base_index_name = index_pattern.format(prefix=message.object_prefix)
        self.elastic = client_for_index(alias)
        self.alias = alias
        self.alias_rollback = alias_rollback
        self.good_imports = 0
        self.errored_imports = 0

    def pre_check(self):
        """Find an available index name to import to

        If an import fails we keep it around for debugging purposes. To allow
        retrying an import we need to select a new index name for each attempt.
        """
        i = 0
        self.index_name = self.base_index_name
        for i in range(10):
            if not self.elastic.indices.exists(self.index_name):
                break
            self.index_name = '{}-{}'.format(self.base_index_name, i)
        else:
            raise ImportFailedException(
                'Could not find an available index name. Last tried: {}'
                .format(self.index_name))
        log.info('Importing to index {}'.format(self.index_name))

    def on_file_available(self, path):
        """Import a file in elasticsearch bulk import format."""
        log.info('Importing from path %s', path)
        with open_by_suffix(path, 'rt') as f:
            good, missing, errors = bulk_import(
                client=self.elastic,
                index=self.index_name,
                doc_type="_doc",
                actions=pair(line.strip() for line in f),
                expand_action_callback=lambda x: x)
            self.good_imports += good
            self.errored_imports += errors

    def on_download_complete(self):
        if self.good_imports == 0 or self.errored_imports > 0:
            # TODO: Delete failed index? Keep for debugging?
            log.critical('Failed import for index %s with %d success and %d failures',
                         self.index_name, self.good_imports, self.errored_imports)
        else:
            def get_alias(name):
                try:
                    return self.elastic.indices.get_alias(name=name).keys()
                except elasticsearch.exceptions.NotFoundError:
                    return []

            old_rollback_aliases = get_alias(self.alias_rollback)
            new_rollback_aliases = get_alias(self.alias)
            self.promote(old_rollback_aliases, new_rollback_aliases)
            self.delete_unused_indices(old_rollback_aliases, new_rollback_aliases)

    def promote(self, old_rollback_aliases, new_rollback_aliases):
        """Promote index to control an alias.

        Uses a two step promotion to hopefully ease rollbacks. When a new index
        is promoted the previous aliases are assigned to a rollback alias. When
        an index is removed from the rollback state it is deleted from the cluster.
        """

        actions = [{'add': {'alias': self.alias, 'index': self.index_name}}]
        for index in old_rollback_aliases:
            actions.append({'remove': {'alias': self.alias_rollback, 'index': index}})
        for index in new_rollback_aliases:
            actions.append({'remove': {'alias': self.alias, 'index': index}})
            actions.append({'add': {'alias': self.alias_rollback, 'index': index}})

        res = self.elastic.indices.update_aliases({'actions': actions})
        if res['acknowledged'] is not True:
            res_str = str(res)
            log.critical("Failed update_aliases: %s", res_str[:1024])
            raise Exception(res_str)
        log.info('Promoted %s to own the %s alias', self.index_name, self.alias)

    def delete_unused_indices(self, old_rollback_aliases, new_rollback_aliases):
        # Don't accidentally delete an alias we still need even if something
        # external has rewritten our aliases to contain duplicates.
        used_indices = {self.index_name}.union(new_rollback_aliases)
        indices_to_delete = set(old_rollback_aliases) - used_indices
        if indices_to_delete:
            res = self.elastic.indices.delete(','.join(indices_to_delete))
            if res['acknowledged'] is not True:
                res_str = str(res)
                log.critical("Failed delete orphaned indices: %s", res_str[:1024])
                raise Exception(res_str)
            log.info('Deleted orphaned indices: %s', ','.join(indices_to_delete))


def run(brokers, client_for_index, topics, group_id):
    log.info('Starting swift daemon')
    swift = SwiftService()
    consumer = kafka.KafkaConsumer(
        bootstrap_servers=brokers,
        group_id=group_id,
        # Commits are manually performed for each batch returned by poll()
        # after they have been processed by elasticsearch.
        enable_auto_commit=False,
        # If we lose the offset safest thing is to reset and
        # wait for the next import. Imports are quite large and
        # we wouldn't want to promote an old import
        auto_offset_reset='latest',
        api_version=mjolnir.kafka.BROKER_VERSION,
        # We kick off large import jobs and wait for them to complete, take in only
        # one at a time.
        max_poll_records=1,
    )

    log.info('Subscribing to: %s', ', '.join(topics))
    consumer.subscribe(topics)
    try:
        while True:
            batch = consumer.poll(timeout_ms=60000)
            # Did the poll time out?
            if not batch:
                continue
            for message in load_and_validate(batch):
                log.info('Received message: %s', str(message))
                try:
                    message.action(client_for_index, swift, message)
                except (ImportFailedException, MalformedUploadException):
                    # If the upload is malformed retrying isn't going to help,
                    # ack and go on with our business. Other errors, such
                    # as connection problems, pass through and cause the
                    # daemon to restart without ack'ing.
                    log.exception('Failed message action')
                else:
                    log.info('Message action complete')
            # Tell kafka we did the work
            offsets = {}
            for tp, records in batch.items():
                offsets[tp] = kafka.OffsetAndMetadata(records[-1].offset + 1, '')
            consumer.commit(offsets)
        log.info('Shutting down swift daemon gracefully')
    finally:
        consumer.close()
        log.info('Exiting swift daemon')
