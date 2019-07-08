from collections import namedtuple
import json
import pytest

from mjolnir.kafka import bulk_daemon

# simplified kafka.consumer.fetcher.ConsumerRecord
ConsumerRecord = namedtuple('ConsumerRecord', ['value'])


def _record(value):
    return ConsumerRecord(json.dumps(value).encode('utf8'))


@pytest.mark.parametrize('expected,records', [
    # standard parse
    [[bulk_daemon.Message('a', 'b', True)], [_record({'container': 'a', 'object_prefix': 'b'})]],
    # swift container must be configured
    [[], [_record({'container': 'unk', 'object_prefix': 'b'})]],
    # Extra fields -> invalid
    [[], [_record({'container': 'a', 'object_prefix': 'b', 'extra': 'c'})]],
    # Missing fields -> invalid
    [[], [_record({'container': 'a'})]],
    [[], [_record({'object_prefix': 'b'})]],
    # Case sensitive
    [[], [_record({'CONTAINER': 'a', 'OBJECT_PREFIX': 'b'})]],

])
def test_load_and_validate(expected, records):
    messages = list(bulk_daemon.load_and_validate({None: records}, {
        'a': True
    }))
    assert messages == expected


def test_pair():
    assert [(0, 1), (2, 3)] == list(bulk_daemon.pair(range(4)))
    assert [(0, 1), (2, 3)] == list(bulk_daemon.pair(range(5)))
    assert [(0, 1), (2, 3), (4, 5)] == list(bulk_daemon.pair(range(6)))


def test_bulk_import(mocker):
    mocker.patch.object(bulk_daemon, 'parallel_bulk').return_value = [
        # Standard response
        (True, {'index': {'result': 'ok'}}),
        # Missing document
        (False, {'index': {'status': 404}}),
        # Failed document (ex: bad script)
        (False, {'index': {'status': 500}}),
    ]
    # Not much left to test after mocking parallel_bulk.
    good, missing, errors = bulk_daemon.bulk_import()
    assert good == 1
    assert missing == 1
    assert errors == 1
