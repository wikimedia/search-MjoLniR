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
from tempfile import TemporaryDirectory

from elasticsearch.helpers import parallel_bulk
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
                except MalformedUploadException:
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
