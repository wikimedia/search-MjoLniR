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
from typing import cast, Any, Callable, Dict, Generic, Iterable, Iterator, \
                   List, Mapping, Optional, Sequence, Tuple, TypeVar, Union

from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk
import elasticsearch.exceptions
import jsonschema
import kafka
import prometheus_client
import requests
from requests.models import Response

from mjolnir.esltr import ModelExistsException, LtrModelUploader, ValidationRequest
import mjolnir.kafka

log = logging.getLogger(__name__)

T = TypeVar('T')
ElasticSupplier = Callable[[str], Elasticsearch]


CONFIG = {
    'search_mjolnir_model': lambda client_for_index, message: ImportLtrModel(  # type: ignore
        client_for_index, message,
    ).run(),
    # key is the name of the swift container files were uploaded to
    'search_popularity_score': lambda client_for_index, message: ImportExistingIndices(  # type: ignore
        client_for_index, message,
        # The main idea here is to send many small-ish bulks
        # in parallel to engage multiple indexing threads of
        # the target indices. Our p50 can be reasonable with
        # a crazy p99, allow a high timeout and let the other
        # threads keep running. Additionally while the docs
        # we are sending are small, they are typically updates
        # to much larger docs that will require time to index.
        thread_count=10,
        # If queue_size < thread_count parallel_bulk can deadlock.
        # https://github.com/elastic/elasticsearch-py/issues/816
        queue_size=10,
        chunk_size=100,
        # Increased to 1 min from default of 10s. No reason we
        # shouldn't wait.  Latency can be checked via logs
        # generated by elasticsearch.
        request_timeout=180
    ).run(),
    'search_glent': lambda client_for_index, message: ImportAndPromote(  # type: ignore
        client_for_index, message,
        # TODO: Should be externally configurable
        index_pattern='glent_{prefix}',
        alias='glent_production',
        alias_rollback='glent_rollback',
        # The clusters are pretty beefy, increase from
        # default of 4.
        thread_count=12
    ).run(),
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
    TIMEOUT = _BULK_ACTION_RESULT.labels(result='timeout')


# namedtuple and jsonschema of incoming requests. This is a subset of
# swift.upload.complete schema v1.0.0 from mediawiki-event-schemas
Message = namedtuple('Message', ('container', 'object_prefix', 'prefix_uri', 'action'))
VALIDATOR = jsonschema.Draft4Validator({
    "type": "object",
    "additionalProperties": True,
    "required": ["swift_container", "swift_object_prefix", "swift_prefix_uri"],
    "properties": {
        "swift_container": {"type": "string"},
        "swift_object_prefix": {"type": "string"},
        "swift_prefix_uri": {"type": "string"}
    }
})

SANITIZE_PATTERN = re.compile('[^a-z0-9]')

# Subset of the model metadata we require to perform model upload
UPLOAD_METADATA_VALIDATOR = jsonschema.Draft4Validator({
    "type": "object",
    "additionalProperties": True,
    "required": ["upload"],
    "properties": {
        "upload": {
            "type": "object",
            "additionalProperties": False,
            "required": ["wikiid", "model_name", "model_type", "feature_definition", "features"],
            "properties": {
                "wikiid": {"type": "string"},
                "model_name": {"type": "string"},
                "model_type": {"type": "string"},
                "feature_definition": {"type": "string"},
                "features": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                # Not required
                "validation_params": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                }
            }
        }
    }
})


def sanitize_index_name(name: str) -> str:
    """Limit the characters allowed into index names for our own sanity"""
    return SANITIZE_PATTERN.sub('_', name.lower())


def load_and_validate(poll_response, config=CONFIG) -> Iterator[Message]:
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
                action = config[value['swift_container']]
            except KeyError:
                Metric.FAIL_NO_CONFIG.inc()
                log.warning("Unknown swift container: %s", value['swift_container'])
                continue

            yield Message(value['swift_container'], value['swift_object_prefix'], value['swift_prefix_uri'], action)


def swift_fetch_prefix_uri(prefix_uri: str) -> Iterator[str]:
    """Fetch swift listing and report available objects

    Parameters
    ----------
    prefix_uri : str
         Absolute url to fetch swift listing

    Yields
    ------
    str
        Absolute url to found swift objects
    """
    # prefix_uri is of the form http://host:port/v1/user/container?prefix=hihihi
    # Grab everything before the ? as the base uri
    container_uri = prefix_uri.split('?', 1)[0]
    res = requests.get(prefix_uri)
    if res.status_code < 200 or res.status_code > 299:
        raise Exception('Failed to fetch swift listing, status_code {}: {}'.format(res.status_code, res.text))
    if res.encoding is None:
        res.encoding = 'utf-8'
    for path in res.iter_lines(decode_unicode=True):
        if path == '':
            continue
        yield os.path.join(container_uri, path)


def swift_download_from_prefix(
    message: Message, abort_on_failure: bool
) -> Iterator[Tuple[str, Response]]:
    """Download files from swift and yield them as they become available

    Parameters
    ----------
    message : Message
    abort_on_failure : bool

    Yields
    ------
    str
        Full uri of the object in swift
    Response
        HTTP response with streamed content
    """
    for file_uri in swift_fetch_prefix_uri(message.prefix_uri):
        with requests.get(file_uri, stream=True) as res:
            if res.status_code < 200 or res.status_code > 299:
                if abort_on_failure:
                    if res.status_code == 404:
                        raise SwiftNotFoundException(file_uri)
                    else:
                        raise Exception('Failed to fetch uri, status code {}: {}'.format(res.status_code, file_uri))
                else:
                    log.warning("Failed fetching from swift, status code %d: %s", res.status_code, file_uri)
                    continue

            yield file_uri, res


def _decode_response_as_text_lines(file_uri: str, res: Response) -> Iterator[str]:
    """Stream requests response into utf8 lines

    Transparently handles decompressing gzip'd content.
    """
    suffix = os.path.splitext(file_uri)[1]
    if suffix.lower() == '.gz':
        # For unknown reasons GzipFile doesn't accept `rt`, requiring
        # to decode ourselves. We also need to strip trailing \n to match
        # iter_lines from requests.
        for line in gzip.GzipFile(mode='r', fileobj=res.raw):
            yield line.decode('utf8').rstrip('\n')
    else:
        if res.encoding is None:
            res.encoding = 'utf-8'
        # This will decode utf8 to str AND strip trailing \n
        yield from res.iter_lines(decode_unicode=True)
    log.info('Finished download of %s', file_uri)


def pair(it: Iterable[T]) -> Iterable[Tuple[T, T]]:
    """Yield pairs of values from the iterable

    Example: pair(range(4)) == ((0,1), (2,3))
    """
    it = iter(it)
    return zip(it, it)


@Metric.BULK_IMPORT.time()
def bulk_import(**kwargs) -> Tuple[int, int, int]:
    """Bulk import data to elasticsearch.

    Tracks bulk import response metrics, reporting both externally to
    prometheus and to the caller.
    """
    log.info('Starting bulk import: {}'.format(str(kwargs)))
    good, missing, errors = 0, 0, 0
    for ok, result in parallel_bulk(raise_on_exception=False, raise_on_error=False, **kwargs):
        action, result = result.popitem()
        status_code = result.get('status', 500)
        if ok:
            good += 1
            try:
                Metric.ACTION_RESULTS[result['result']].inc()
            except KeyError:
                Metric.OK_UNKNOWN.inc()
        elif status_code == 'TIMEOUT':
            Metric.TIMEOUT.inc()
            errors += 1
        elif not isinstance(status_code, int):
            # Previously found TIMEOUT status_code here
            Metric.FAILED.inc()
            log.warning(
                'Failed bulk %s request with invalid status_code %s: %s',
                action, str(status_code), str(result)[:1024])
            errors += 1
        elif status_code == 404:
            # 404 are quite common so we log them separately. The analytics
            # side doesn't know the namespace mappings and attempts to send all
            # updates to <wiki>_content, letting the docs that don't exist fail
            missing += 1
            Metric.MISSING.inc()
        elif status_code >= 400 and status_code < 500:
            # Bulk contained invalid records, can't do much beyond logging
            Metric.FAILED.inc()
            log.warning('Failed bulk %s request: %s', action, str(result)[:1024])
            errors += 1
        elif status_code >= 500 and status_code < 600:
            # primary not available, etc. Internal elasticsearch errors. Should be retryable
            raise Exception(
                "Internal elasticsearch error on {}, status code {}: {}".format(action, status_code, str(result)))
        else:
            raise Exception(
                "Unexpected response on {}, status code {}: {}".format(action, status_code, str(result)))

    log.info('Completed import with %d success %d missing and %d errors', good, missing, errors)
    return good, missing, errors


def expand_string_actions(pair: Tuple[Union[Mapping, str], str]) -> Tuple[Mapping, str]:
    """Expand meta in a (meta, doc) bulk import pair

    Passing already encoded strings through the elasticsearch bulk helpers generally works,
    but when reporting failures the _process_bulk_chunk helper tries to determine the op_type
    and fails due to the following. Decode the string as necessary so error handling works.
        op_type, action = data[0].copy().popitem()
    """
    if isinstance(pair[0], str):
        return json.loads(pair[0]), pair[1]
    else:
        return cast(Tuple[Mapping, str], pair)


class Peekable(Generic[T]):
    """Wrap an iterable to allow peeking at the next value"""
    def __init__(self, it: Iterator[T]) -> None:
        self.it = it
        self.cache = cast(Optional[T], None)

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        if self.cache is None:
            return next(self.it)
        else:
            value = self.cache
            self.cache = None
            return value

    def peek(self) -> T:
        if self.cache is None:
            self.cache = next(self.it)
        return self.cache


class ImportFailedException(Exception):
    """The import has failed and must not be retried"""
    pass


class MalformedUploadException(ImportFailedException):
    """A file downloaded from swift cannot be processed"""
    pass


class SwiftNotFoundException(ImportFailedException):
    """An object requested from swift returned 404"""
    pass


class UploadAction:
    """Perform an action based on a new file upload(s) becoming available"""
    abort_on_failure = True

    def __init__(self, client_for_index: ElasticSupplier, message: Message) -> None:
        self.client_for_index = client_for_index
        self.message = message

    @Metric.PROCESS_MESSAGE.time()
    def run(self) -> None:
        self.pre_check()
        for uri, response in swift_download_from_prefix(self.message, self.abort_on_failure):
            try:
                self.on_file_available(uri, response)
            except ImportFailedException:
                if self.abort_on_failure:
                    raise
                log.exception('Failed processing file %s', uri)
        self.on_download_complete()

    def pre_check(self) -> None:
        pass

    def on_file_available(self, uri: str, response: Response) -> None:
        raise NotImplementedError()

    def on_download_complete(self) -> None:
        pass


class ImportExistingIndices(UploadAction):
    """
    Import file(s) to existing indices spread across multiple clusters.
    Imported file(s) must specify both index name and doc type. A single
    file must not contain updates for multiple indices.
    """

    def __init__(self, client_for_index: ElasticSupplier, message: Message, **kwargs) -> None:
        super(ImportExistingIndices, self).__init__(client_for_index, message)
        self.bulk_kwargs = kwargs

    # If individual files somehow fail that is unfortunate, but keep processing the remaining
    # files. TODO: Some sort of circuit breaker that doesn't send 100M invalid updates when
    # invalid files are published might be nice.
    abort_on_failure = False

    def on_file_available(self, uri: str, response: Response) -> None:
        # Metadata we aren't interested in
        if os.path.basename(uri)[0] == '_':
            return
        lines = Peekable(_decode_response_as_text_lines(uri, response))
        try:
            header = json.loads(lines.peek())
            action, meta = header.popitem()
            index_name = meta['_index']
        except (ValueError, KeyError):
            raise MalformedUploadException(
                "Loaded file is malformed and cannot be processed: {}".format(uri))

        # Ignoring errors, can't do anything useful with them. They still
        # get logged and counted.
        bulk_import(client=self.client_for_index(index_name),
                    actions=pair(line.strip() for line in lines),
                    expand_action_callback=expand_string_actions,
                    **self.bulk_kwargs)


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
    def __init__(
        self, client_for_index: ElasticSupplier, message: Message,
        index_pattern: str, alias: str, alias_rollback: str, **kwargs
    ) -> None:
        super(ImportAndPromote, self).__init__(client_for_index, message)
        self.base_index_name = index_pattern.format(prefix=message.object_prefix)
        self.elastic = client_for_index(alias)
        self.alias = alias
        self.alias_rollback = alias_rollback
        self.bulk_kwargs = kwargs
        self.good_imports = 0
        self.errored_imports = 0

    def pre_check(self) -> None:
        """Find an available index name to import to

        If an import fails we keep it around for debugging purposes. To allow
        retrying an import we need to select a new index name for each attempt.
        """
        i = 0
        self.index_name = self.base_index_name
        if self.elastic.indices.exists(self.index_name):
            for i in range(10):
                self.index_name = '{}-{}'.format(self.base_index_name, i)
                if not self.elastic.indices.exists(self.index_name):
                    break
            else:
                raise ImportFailedException(
                    'Could not find an available index name. Last tried: {}'
                    .format(self.index_name))
        log.info('Importing to index {}'.format(self.index_name))

    def on_file_available(self, uri: str, response: Response) -> None:
        # Metadata we aren't interested in
        if os.path.basename(uri)[0] == '_':
            return
        lines = _decode_response_as_text_lines(uri, response)
        """Import a file in elasticsearch bulk import format."""
        log.info('Importing from uri %s', uri)
        good, missing, errors = bulk_import(
            client=self.elastic,
            index=self.index_name,
            doc_type="_doc",
            actions=pair(line.strip() for line in lines),
            expand_action_callback=expand_string_actions,
            **self.bulk_kwargs)
        self.good_imports += good
        self.errored_imports += errors

    def on_download_complete(self) -> None:
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

    def promote(self, old_rollback_aliases: List[str], new_rollback_aliases: List[str]) -> None:
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

    def delete_unused_indices(self, old_rollback_aliases: List[str], new_rollback_aliases: List[str]) -> None:
        """Delete indices that are no longer necessary

        As new indices are promoted old indices become unused. Delete the
        unused indices to keep cruft from building up

        Parameters
        ----------
        old_rollback_aliases : list of str
        new_rollback_aliases : list of str
        """
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


class ImportLtrModel(UploadAction):
    def __init__(
        self, client_for_index: ElasticSupplier, message: Message
    ) -> None:
        super().__init__(client_for_index, message)
        self.downloaded_files = cast(Dict[str, str], {})

    def on_file_available(self, uri: str, response: Response) -> None:
        # We are going to download everything from swift into the process, and
        # then send it out after recieving everything. This should be under
        # 100M and not a big deal.
        suffix = os.path.splitext(uri)[1]
        if suffix.lower() != '.json':
            # We are only interested in a pair of json files
            # but the upload contains binary xgb as well.
            return

        basename = os.path.basename(uri)
        if basename in self.downloaded_files:
            raise ImportFailedException('Duplicate files: {}'.format(uri))
        self.downloaded_files[basename] = '\n'.join(
            _decode_response_as_text_lines(uri, response))

    def validate_input(self) -> Tuple[Sequence[Mapping], Mapping[str, Any], List[str]]:
        model = None
        metadata = None
        errors = []
        try:
            model = json.loads(self.downloaded_files['model.json'])
        except KeyError as e:
            errors.append('Missing model.json: {}'.format(e))
        except json.decoder.JSONDecodeError as e:
            errors.append('Unable to parse model.json: {}'.format(e))

        try:
            # TODO: _METADATA.JSON doesn't exist in mjolnir,only discovery-analytics.
            metadata = json.loads(self.downloaded_files['_METADATA.JSON'])
        except KeyError as e:
            errors.append('Missing _METADATA.JSON: {}'.format(e))
        except json.decoder.JSONDecodeError as e:
            errors.append('Unable to parse _METADATA.JSON: {}'.format(e))
        else:
            metadata_errors = list(UPLOAD_METADATA_VALIDATOR.iter_errors(metadata))
            if metadata_errors:
                errors.extend(metadata_errors)

        return model, metadata, errors  # type: ignore

    def on_download_complete(self) -> None:
        model, metadata, errors = self.validate_input()
        if errors:
            raise ImportFailedException('Invalid model upload request: {}'.format(', '.join(errors)))

        upload = metadata['upload']
        validation = None
        if 'validation_params' in upload:
            validation = ValidationRequest(upload['wikiid'], upload['validation_params'])

        try:
            LtrModelUploader(
                self.client_for_index(upload['wikiid']),
                upload['model_name'],
                upload['model_type'],
                model,
                upload['feature_definition'],
                upload['features'],
                validation
            ).upload()
        except ModelExistsException:
            raise ImportFailedException('Cannot process message, model already exists')


def run(brokers: str, client_for_index: ElasticSupplier, topics: List[str], group_id: str) -> None:
    log.info('Starting swift daemon')
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
        # Our jobs take quite awhile to run, increase from default of 5 min
        # to 60 min to account for the long running imports. Expect most messages
        # to process in < 10 min each. Expect glent messages to take up to 45min
        # to process. This is a complete hack at this point, we shouldn't be
        # using hour long poll interval timeouts. This means if a node disapears
        # it will take an hour to notice.
        max_poll_interval_ms=1000 * 60 * 60,
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
                    message.action(client_for_index, message)
                except ImportFailedException:
                    # If the import failed retrying isn't going to help,
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
