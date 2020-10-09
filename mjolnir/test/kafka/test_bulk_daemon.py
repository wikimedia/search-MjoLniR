from collections import namedtuple
from io import BytesIO
import gzip
import json

import pytest
import requests
from unittest.mock import Mock

from mjolnir.kafka import bulk_daemon

# simplified kafka.consumer.fetcher.ConsumerRecord
ConsumerRecord = namedtuple('ConsumerRecord', ['value'])


def _record(value):
    return ConsumerRecord(json.dumps(value).encode('utf8'))


def test_sanitize_index_name():
    assert 'foo_bar' == bulk_daemon.sanitize_index_name('foo/bar')
    assert '20190101' == bulk_daemon.sanitize_index_name('20190101')


@pytest.mark.parametrize('expected,records', [
    # standard parse
    [[bulk_daemon.Message('a', 'b', 'c', True)],
     [_record({'swift_container': 'a', 'swift_object_prefix': 'b', 'swift_prefix_uri': 'c'})]],
    # swift container must be configured
    [[], [_record({'container': 'unk', 'object_prefix': 'b', 'swift_prefix_uri': 'c'})]],
    # Extra fields -> invalid
    [[], [_record({'container': 'a', 'object_prefix': 'b', 'extra': 'c', 'swift_prefix_uri': 'c'})]],
    # Missing fields -> invalid
    [[], [_record({'container': 'a', 'swift_prefix_uri': 'c'})]],
    [[], [_record({'object_prefix': 'b', 'swift_prefix_uri': 'c'})]],
    # Case sensitive
    [[], [_record({'CONTAINER': 'a', 'OBJECT_PREFIX': 'b', 'SWIFT_PREFIX_URI': 'c'})]],

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
        (True, {'index': {'result': 'ok'}}),
        # Missing document
        (False, {'index': {'status': 404}}),
        # Failed document (ex: bad script)
        (False, {'index': {'status': 400}}),
    ]
    # Not much left to test after mocking parallel_bulk.
    good, missing, errors = bulk_daemon.bulk_import()
    assert good == 1
    assert missing == 1
    assert errors == 1


@pytest.mark.parametrize('expected_index_name,exists', [
    ('test_prefix', set()),
    ('test_prefix-0', {'test_prefix'}),
    ('test_prefix-1', {'test_prefix', 'test_prefix-0'}),
    (None, {'test_prefix'}.union('test_prefix-{}'.format(i) for i in range(10))),
])
def test_ImportAndPromote_pre_check(expected_index_name, exists):
    elastic = Mock()
    elastic.indices.exists.side_effect = lambda name: name in exists

    message = bulk_daemon.Message('test_container', 'prefix', 'http://fake', None)
    action = bulk_daemon.ImportAndPromote(lambda x: elastic, message, 'test_{prefix}', 'a', 'b')
    try:
        action.pre_check()
    except bulk_daemon.ImportFailedException:
        assert expected_index_name is None
    else:
        assert action.index_name == expected_index_name


@pytest.mark.parametrize('expect_actions,expect_delete,aliases', [
    # First run, no aliases exist
    [
        # expected actions
        [{'add': {'alias': 'alias', 'index': 'test_pickles'}}],
        # expected deletes
        False,
        # existing aliases
        {}
    ],
    # Move old alias to rollback
    [
        # expected actions
        [
            {'add': {'alias': 'alias', 'index': 'test_pickles'}},
            {'remove': {'alias': 'alias', 'index': 'test_old'}},
            {'add': {'alias': 'alias_rollback', 'index': 'test_old'}}
        ],
        # expected deletes
        False,
        # existing aliases
        {'alias': ['test_old']}
    ],
    # Move old alias, delete oldest alias
    [
        # expected actions
        [
            {'add': {'alias': 'alias', 'index': 'test_pickles'}},
            {'remove': {'alias': 'alias_rollback', 'index': 'test_rollback'}},
            {'remove': {'alias': 'alias', 'index': 'test_old'}},
            {'add': {'alias': 'alias_rollback', 'index': 'test_old'}}
        ],
        # expected deletes
        'test_rollback',
        # existing aliases
        {'alias': ['test_old'], 'alias_rollback': ['test_rollback']}
    ]
])
def test_promotion(expect_actions, expect_delete, aliases):
    elastic = Mock()
    state = {
        'alias': False,
        'delete': False,
    }

    def get_alias(name):
        x = aliases.get(name, [])
        return dict(zip(x, [None] * len(x)))

    def update_aliases(actions):
        assert state['alias'] is False
        state['alias'] = True
        assert actions['actions'] == expect_actions
        return {'acknowledged': True}

    def delete(indices):
        assert state['delete'] is False
        state['delete'] = True
        assert indices == expect_delete
        return {'acknowledged': True}

    elastic.indices.exists.side_effect = lambda name: False
    elastic.indices.get_alias.side_effect = get_alias
    elastic.indices.update_aliases = update_aliases
    elastic.indices.delete = delete

    message = bulk_daemon.Message('test_container', 'pickles', 'http://fake', None)
    action = bulk_daemon.ImportAndPromote(lambda x: elastic, message, 'test_{prefix}', 'alias', 'alias_rollback')
    action.pre_check()
    action.good_imports = 1
    action.on_download_complete()
    assert state['alias'] is True
    if not expect_delete:
        assert state['delete'] is False


def text_response(url, str_content):
    res = requests.Response()
    res.status_code = 200
    res.encoding = 'utf8'
    res.raw = BytesIO(str_content.encode('utf8'))
    res.url = url
    return res


def gzip_response(url, str_content):
    gzip_bytes = BytesIO()
    with gzip.GzipFile(fileobj=gzip_bytes, mode='w') as f:
        f.write(str_content.encode('utf8'))
    gzip_bytes.seek(0)

    res = requests.Response()
    res.url = url + '.gz'
    res.status_code = 200
    res.encoding = None
    res.raw = gzip_bytes
    return res


@pytest.mark.parametrize('encode', [text_response, gzip_response])
def test_decode_text_response_as_text_lines(encode):
    lines = ['a', 'b', 'c']
    response = encode('http://a.b/c.txt', '\n'.join(lines))

    # TODO: Why do we pass response.url separately?
    decoded = list(bulk_daemon._decode_response_as_text_lines(response.url, response))
    assert lines == decoded
