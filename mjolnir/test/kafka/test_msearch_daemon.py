import json

import pytest
import requests

from mjolnir.kafka.msearch_daemon import Daemon, FlexibleInterval, MetricMonitor, StreamingEMA


def test_consume_nothing(mocker):
    mocker.patch('kafka.KafkaProducer')
    # Test it doesn't blow up
    Daemon(None).consume([])


def test_consume_end_sigil(mocker, monkeypatch):
    # simple mock we can observe
    monkeypatch.setattr('kafka.KafkaProducer', MockProducer)
    # Don't let reflect sit around forever
    monkeypatch.setattr(Daemon, 'REFLECT_WAIT', 0.01)
    # Fetching result offsets uses the consumer
    mock = mocker.patch('kafka.KafkaConsumer')
    mock.partitions_for_topic.return_value = [0]
    mock.position.return_value = 42
    # Finally we can run something
    daemon = Daemon(None)
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
    daemon = Daemon(None)
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


def test_streaming_ema():
    # Starts in a not-ready state
    now = 0
    alpha = 0.1
    ema = StreamingEMA(alpha=alpha, max_sec_valid=60, clock=lambda: now)
    assert not ema.is_below_threshold(100)
    # Providing a single value isn't enough to initialize the state
    ema.update(42)
    assert not ema.is_below_threshold(100)
    # Providing a second value gets everything going
    now += 1
    ema.update(43)
    assert ema.is_below_threshold(100)
    assert ema.value == 1
    # Simple check of ema calculation
    now += 1
    ema.update(45)
    assert ema.value == (alpha * 2) + ((1 - alpha) * 1)
    # Waiting multiple seconds between updates spreads the count out.
    # 200 ops in 10 seconds, 20ops/s.
    now += 10
    ema.update(245)
    assert ema.value == pytest.approx(13.41, abs=0.01)
    assert ema.is_below_threshold(100)
    # The metric is only valid without updates for max_sec_valid
    now += 60
    assert not ema.is_valid
    assert not ema.is_below_threshold(100)
    # Updating with large values moves above threshold
    ema.update(10000)
    assert ema.is_valid
    assert not ema.is_below_threshold(100)
    # When the counter resets (shard move?) we do not become valid
    # and the metric increases.
    now += 15
    old_value = ema.value
    ema.update(0)
    assert ema.is_valid
    assert ema.value > old_value
    assert not ema.is_below_threshold(100)
    # We can transition back below threshold
    now += 60
    ema.update(60)
    assert ema.is_valid
    assert ema.is_below_threshold(100)


class LatestValue(object):
    is_valid = False
    value = None

    def is_below_threshold(self, threshold):
        return self.is_valid and self.value < threshold

    def update(self, value):
        self.value = value
        self.is_valid = True


def test_prod_load_monitor(mocker):
    mocker.patch('mjolnir.kafka.msearch_daemon.FlexibleInterval')
    stat = 1000
    monitor = MetricMonitor(lambda: stat, LatestValue(), threshold=916)
    # Starts in a disabled state
    assert not monitor.is_below_threshold
    # Transition to valid state
    stat = 408
    monitor.update_metric()
    assert monitor.is_below_threshold
    # Transition back to disabled state
    stat = 1000
    monitor.update_metric()
    assert not monitor.is_below_threshold
    # New monitor with values above threshold does not activate
    monitor = MetricMonitor(lambda: stat, LatestValue(), threshold=917)
    monitor.update_metric()
    assert not monitor.is_below_threshold


def test_flexible_interval():
    now = 0

    def clock():
        return now
    interval = FlexibleInterval(min_value=20, max_value=200, ratio=0.5, clock=clock)
    assert interval.value == 20
    interval.decrease()
    assert interval.value == 20
    interval.increase()
    assert interval.value == 30
    interval.increase()
    assert interval.value == 45
    interval.value = 200
    interval.increase()
    assert interval.value == 200
    interval.decrease()
    assert interval.value == 20
