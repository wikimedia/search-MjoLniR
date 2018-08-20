import json

import requests

from mjolnir.kafka import msearch_daemon


def test_consume_nothing(mocker):
    mocker.patch('kafka.KafkaProducer')
    # Test it doesn't blow up
    daemon = msearch_daemon.Daemon(None)
    daemon.consume([])


def test_consume_end_sigil(mocker, monkeypatch):
    # simple mock we can observe
    monkeypatch.setattr('kafka.KafkaProducer', MockProducer)
    # Don't let reflect sit around forever
    monkeypatch.setattr(msearch_daemon.Daemon, 'REFLECT_WAIT', 0.01)
    # Fetching result offsets uses the consumer
    mock = mocker.patch('kafka.KafkaConsumer')
    mock.partitions_for_topic.return_value = [0]
    mock.position.return_value = 42
    # Finally we can run something
    daemon = msearch_daemon.Daemon(None)
    daemon.consume([
        {'run_id': 'abc', 'meta': {}, 'complete': True, 'partition': 0}
    ])
    assert len(daemon.producer.sent) == 0
    assert len(daemon.ack_all_producer.sent) == 1
    sent_topic, sent_message = daemon.ack_all_producer.sent[0]
    sent_message = json.loads(sent_message.decode('utf8'))
    assert sent_message['run_id'] == 'abc'


def test_consume_msearch_req(mocker, monkeypatch):
    # Mock out the actual search request
    make_req_mock = mocker.patch('mjolnir.cirrus.make_request')
    make_req_mock.return_value = requests.models.Response()
    # simple mock we can observe
    monkeypatch.setattr('kafka.KafkaProducer', MockProducer)
    daemon = msearch_daemon.Daemon(None)
    daemon.consume([
        {'run_id': 'zyx', 'meta': {}, 'request': 'invalid unit test req'}
    ])
    assert len(daemon.producer.sent) == 1
    assert len(daemon.ack_all_producer.sent) == 0
    sent_topic, sent_message = daemon.producer.sent[0]
    sent_message = json.loads(sent_message.decode('utf8'))
    assert sent_message['run_id'] == 'zyx'


class MockFuture(object):
    def add_errback(self, fn):
        pass

    def get(self):
        pass


class MockProducer(object):
    def __init__(self, *args, **kwargs):
        self.sent = []

    def send(self, topic, message):
        self.sent.append((topic, message))
        return MockFuture()

    def close(self):
        pass

    def flush(self):
        pass
