"""
Daemon for collecting elasticsearch queries from a kafka topic, running them,
and pushing the results into a second kafka topic. This runs on the production
side of the network to have access to relforge servers.
"""

import json
import logging
import multiprocessing.dummy
import queue
import threading
import time

import elasticsearch
from elasticsearch import Elasticsearch
import kafka
import kafka.common
import kafka.consumer.subscription_state
import prometheus_client
import requests

import mjolnir.cirrus
import mjolnir.kafka


log = logging.getLogger(__name__)


class Metric(object):
    """A namespace for our runtime metrics"""
    RECORDS_PROCESSED = prometheus_client.Counter(
        'mjolnir_msearch_records_total',
        'Number of kafka records processed')
    INTERVAL_VALUE = prometheus_client.Gauge(
        'mjolnir_msearch_interval_sec',
        'Seconds between polling elasticsearch for qps stats')
    EMA = prometheus_client.Gauge(
        'mjolnir_msearch_ema_qps',
        'Local estimate of canary index qps')
    PROCESS_BATCH = prometheus_client.Summary(
        'mjolnir_msearch_process_batch_seconds',
        'Time taken to process a batch of records from kafka')


def iter_queue(queue):
    """Yield items from a queue"""
    while True:
        record = queue.get()
        try:
            # Queue is finished, nothing more will arrive
            if record is None:
                return
            yield record
        finally:
            queue.task_done()


class Daemon(object):
    def __init__(self, brokers, n_workers=5, topic_work=mjolnir.kafka.TOPIC_REQUEST,
                 topic_result=mjolnir.kafka.TOPIC_RESULT, topic_complete=mjolnir.kafka.TOPIC_COMPLETE,
                 max_request_size=4*1024*1024, query_total_threshold=100, prometheus_port=9161,
                 group_id='mjolnir_msearch', max_concurrent_searches=1):
        """Initialize the msearch daemon

        Parameters
        ----------
        brokers : list of str
            Brokers to use to bootstrap kafka cluster access
        n_workers : int
            The number of requests to issue to elasticsearch in parallel
        topic_work : str
            Kafka topic to read msearch requests from
        topic_result : str
            Kafka topic to write msearch results to
        topic_complete : str
            Kafka topic for sending control messages
        max_request_size : int
            The maximum number of bytes to send in a single kafka produce request
        query_total_threshold : int
            A threshold for the number of shard queries per second to the enwiki_content
            full_text stats group.  When above this threshold the daemon will not consume
            records and will not generate load on the cluster.
        prometheus_port : int
            Port to expose prometheus metrics collection at
        group_id : str
            Kafka consumer group to join
        max_concurrent_searches : int
            The maximum number of search requests within a single msearch to run in parallel. The
            total concurrent queries possible to issue across a deployment will be:
                    max_concurrent_searches * n_workers * n_kafka_partitions * n_elastic_shards
        """
        self.brokers = brokers
        self.n_workers = n_workers
        self.topic_work = topic_work
        self.topic_result = topic_result
        self.topic_complete = topic_complete
        self.prometheus_port = prometheus_port
        self.group_id = group_id
        self.max_concurrent_searches = max_concurrent_searches
        # Standard producer for query results
        self.producer = kafka.KafkaProducer(bootstrap_servers=brokers,
                                            max_request_size=max_request_size,
                                            compression_type='gzip',
                                            api_version=mjolnir.kafka.BROKER_VERSION)
        # More reliable producer for reflecting end run sigils. As this
        # is only used for sigils and not large messages like es responses
        # compression is unnecessary here.
        self.ack_all_producer = kafka.KafkaProducer(bootstrap_servers=brokers,
                                                    acks='all',
                                                    api_version=mjolnir.kafka.BROKER_VERSION)
        # No clue how many is appropriate. n_workers seems reasonable enough.
        # We want enough to keep the workers busy, but not so many that the
        # commited offsets are siginficantly ahead of the work actually being
        # performed.
        self.work_queue = queue.Queue(n_workers)
        # Toggle our kafka subscription on/off based on qps of a canary index
        self.load_monitor = MetricMonitor.es_query_total(
                Elasticsearch(), 'enwiki_content', 'full_text',
                threshold=query_total_threshold)

    def run(self):
        prometheus_client.start_http_server(self.prometheus_port)
        try:
            while True:
                # If we are seeing production traffic do nothing and sit around.
                self.load_monitor.wait_until_below_threshold()

                # No production traffic, go ahead and subscribe to kafka
                self.consume(self.iter_records())
        finally:
            self.producer.close()
            self.ack_all_producer.close()

    def iter_records(self):
        consumer = kafka.KafkaConsumer(bootstrap_servers=self.brokers,
                                       group_id='mjolnir_msearch',
                                       enable_auto_commit=False,
                                       auto_offset_reset='latest',
                                       value_deserializer=lambda x: json.loads(x.decode('utf8')),
                                       api_version=mjolnir.kafka.BROKER_VERSION,
                                       # Msearch requests are relatively heavy at a few tens of ms each.
                                       # 50 requests at 50ms each gives us ~2.5s to process a batch. We
                                       # keep this low so kafka regularly gets re-pinged.
                                       max_poll_records=min(500, 50 * self.n_workers))
        consumer.subscribe([self.topic_work])
        try:
            last_commit = 0
            offset_commit_interval_sec = 60
            offsets = {}
            while self.load_monitor.is_below_threshold:
                now = time.monotonic()
                if offsets and now - last_commit > offset_commit_interval_sec:
                    consumer.commit_async(offsets)
                    last_commit = now
                    offsets = {}
                # By polling directly, rather than using the iter based api, we
                # have the opportunity to regularly re-check the load monitor
                # and transition out of the consuming state if needed.
                poll_response = consumer.poll(timeout_ms=60000)
                if not poll_response:
                    continue
                with Metric.PROCESS_BATCH.time():
                    for tp, records in poll_response.items():
                        for record in records:
                            self.load_monitor.notify()
                            yield record.value
                    # Wait for all the work to complete
                    self.work_queue.join()
                for tp, records in poll_response.items():
                    offsets[tp] = kafka.OffsetAndMetadata(records[-1].offset + 1, '')
                Metric.RECORDS_PROCESSED.inc(sum(len(x) for x in poll_response.values()))
        finally:
            if offsets:
                consumer.commit(offsets)
            consumer.close()

    def consume(self, records):
        def work_fn():
            self._handle_records(iter_queue(self.work_queue))

        # Use the dummy pool since these workers will primarily wait on elasticsearch
        worker_pool = multiprocessing.dummy.Pool(self.n_workers, work_fn)
        try:
            for record in records:
                if 'complete' in record:
                    # This is handled directly, rather than queued, because the
                    # consumer guarantees the offset won't be commited until the
                    # next record is consumed. By not consuming any more records
                    # we guarantee at least once processing of these sigils.
                    self._reflect_end_run(record)
                else:
                    self.work_queue.put(record)
        except KeyboardInterrupt:
            # Simply exit the work loop, let everything clean up as expected.
            pass
        finally:
            worker_pool.close()
            for i in range(self.n_workers):
                self.work_queue.put(None)
            worker_pool.join()

        # It is possible, if some workers have errors, for the queue to not be
        # completely emptied. Make sure it gets finished
        if self.work_queue.qsize() > 0:
            log.warning('Work queue not completely drained on shut down. Draining')
            # We call repeatedly because the None values exit the iterator
            while self.work_queue.qsize() > 0:
                work_fn()

    def _reflect_end_run(self, record):
        """Reflect and end run sigil into the complete topic

        This is handled directly in the consumer thread, rather than as part of the
        work queue, to ensure that the offset is not committed to kafka until after
        processing is completed and it has been sucessfully reflected.

        Parameters
        ----------
        record : dict
           Deserialized end run sigil
        """
        log.info('reflecting end sigil for run %s and partition %d' %
                 (record['run_id'], record['partition']))
        # Wait for everything to at least start processing. We don't
        # actually know when the workers are finally idle.
        self.work_queue.join()
        future = self.ack_all_producer.send(
            self.topic_complete, json.dumps(record).encode('utf8'))
        future.add_errback(lambda e: log.critical(
            'Failed to send the "end run" message: %s', e))
        # Wait for ack (or failure to ack)
        future.get()

    def _handle_records(self, work):
        """Handle a single kafka record from request topic

        Parameters
        ----------
        work : iterable of dict
            Yields deserialized requests from kafka to process
        """
        # Standard execution of elasticsearch bulk query
        session = requests.Session()
        for record in work:
            try:
                response = mjolnir.cirrus.make_request(
                    'msearch', session, ['http://localhost:9200'], record['request'], reuse_url=True,
                    query_string={'max_concurrent_searches': self.max_concurrent_searches})
                future = self.producer.send(self.topic_result, json.dumps({
                    'run_id': record['run_id'],
                    'meta': record['meta'],
                    'status_code': response.status_code,
                    'text': response.text,
                }).encode('utf8'))
                future.add_errback(lambda e: log.error(
                    'Failed to send a message to the broker: %s', e))
            except:  # noqa: E722
                log.exception('Exception processing record')


class FlexibleInterval(object):
    """Interval that scales up and down

    Interval that increases in length unless regularly reset. When the interval
    is being regularly reset due to activity it stays at the minimum value.
    Over time when not reset it tends to the max value.  This allows for quick
    updates when necessary, and throttled behavior when not needed.

    With default settings it takes 8 steps to go from min to max.

    Parameters
    ----------
    min_value : int
        Value taken when the interval is reset, in seconds
    max_value : int
        The maximum value the interval can take, in seconds
    ratio : float
        A value giving the ratio to increase interval by after
        each iteration.
    clock : callable
        0 arity function returning the current time in seconds
    """

    def __init__(self, min_value=15, max_value=600, ratio=0.4, clock=time.monotonic):
        self.min_value = min_value
        self.max_value = max_value
        self.ratio = ratio
        self.clock = clock
        self.value = self.min_value
        Metric.INTERVAL_VALUE.set_function(lambda: self.value)

    def decrease(self):
        self.value = self.min_value

    def increase(self):
        next_value = self.value * (1 + self.ratio)
        self.value = min(self.max_value, next_value)

    def start(self, fn):
        stopped = threading.Event()

        def inner():
            now = self.clock()
            stop_at = now + self.value
            # Wait no more than min_value to allow quick transition
            # from max_value to min_value
            while not stopped.wait(min(self.min_value, stop_at - now)):
                now = self.clock()
                # If the interval has reset from a large value we must reset
                # stop_at as well
                if stop_at > now + self.value:
                    stop_at = now
                elif now < stop_at:
                    continue
                fn()
                stop_at = now + self.value
                # We post-increase, or we would never hit the min_value
                self.increase()
        threading.Thread(target=inner, daemon=True).start()
        return stopped.set


class StreamingEMA(object):
    """Maintain a streaming exponential moving average

    Accepts a total count of operations performed since an unknown time and
    converts it into an estimation of the number of operations per second
    exponentially weighted towards recent values.

    Since this operates on per-second granularity the influence of a particular
    second is (1 - alpha) ** age_in_s. For example with an alpha of 0.1 a value
    30s out will have an influence of 4.2%, and a value 60s out will have an
    0.2% influence on the ema.

    Parameters
    ----------
    alpha : float in domain [0, 1]
        Per-second exponential decay
    max_sec_valid : int
        Number of seconds ema is considered valid without new data. This must be
        larger than the maximum update interval, or the metric will be invalid
        when transitioning from low volume to high volume updates.
    """

    def __init__(self, alpha=0.1, max_sec_valid=660, clock=time.monotonic):
        self.alpha = alpha
        self.clock = clock
        self.max_sec_valid = max_sec_valid
        self.count = 0
        self.prev_count = None
        self.prev_time = None
        self.value = None
        Metric.EMA.set_function(lambda: self.value)

    @property
    def is_valid(self):
        return self.value is not None \
                and self.clock() < self.prev_time + self.max_sec_valid

    def is_below_threshold(self, threshold):
        return self.is_valid and self.value < threshold

    def update(self, count):
        """Update EMA estimation

        Parameters
        ----------
        count : int
            A count of the number of operations that have occured. Each call
            should receive a value >= the previous one. If the value
            decreases the operations per second for that time period are
            assumed to be 125% the current ema.
        """
        now = self.clock()
        if self.prev_count is not None:
            time_delta, per_second = self._calc_deltas(now, count)
            if per_second is None:
                # Not enough information to determine per-second values.
                # Since we don't use this info dont update self.prev_*
                return
            elif self.value is None:
                # Initialize ema to the first value we find
                self.value = per_second
            else:
                # Run forward each second of ema. Not sure flooring
                # with a cast is exactly correct, but probably works
                # well enough
                for _ in range(int(time_delta)):
                    self.value = (per_second * self.alpha) + \
                            (self.value * (1 - self.alpha))
        # Track our previous values so we can calculate the next
        self.prev_count = count
        self.prev_time = now

    def _calc_deltas(self, new_time, new_count):
        time_delta = new_time - self.prev_time
        per_second = None
        # If data comes in too fast ignore it
        if time_delta >= 1:
            count_delta = new_count - self.prev_count
            if count_delta >= 0:
                # Convert the count into a per second value.
                per_second = count_delta / time_delta
            elif self.value is not None:
                # Delta could be negative when a shard moves. Since we use this to
                # stop doing things if the value gets too high, safest thing to do
                # when we don't know seems to be to increase from the last delta. A
                # series of failures (not sure how) will then trip the check.
                per_second = self.value * 1.25
        return time_delta, per_second


class MetricMonitor(object):
    """Monitors a metric and triggers an event when it goes below a threshold.

    Provides values to a streaming metric on irregular intervals. After providing
    a value checks if the metric is below a threshold.
    """
    def __init__(self, fetch_stat, metric, threshold=50, interval=FlexibleInterval()):
        self.fetch_stat = fetch_stat
        self.metric = metric
        self.threshold = threshold
        # Starts in the False state
        self._is_below_threshold = threading.Event()
        self.interval = interval
        # Calling _stop will stop the update interval. Only for pytest.
        self._stop = interval.start(self.update_metric)

    @classmethod
    def es_query_total(cls, cluster, index, group, **kwargs):
        def fetch_stat():
            try:
                response = cluster.indices.stats(index=index, groups=[group], metric='search')
            except elasticsearch.NotFoundError:
                # If our index doesn't exist we can't possibly allow things
                # to continue. Report the metric unavailable and wait for
                # the index to exist.
                log.exception('Index not found while fetching index stats for %s', index)
                return None
            except elasticsearch.TransportError:
                # Connection error to elasticsearch, could be network, restarts, etc.
                log.exception('Transport error while fetching index stats for %s', index)
                return None

            try:
                query_total = response['_all']['total']['search']['groups'][group]['query_total']
                log.debug('Group %s in index %s reported query_total of %d', group, index, query_total)
                return query_total
            except KeyError:
                # Typically this means the group hasn't collected any stats.
                # This could happen after a full cluster restart but before any
                # prod traffic is run through. I'm a bit wary of always
                # returning 0, but it is correct.
                log.info('No stats in index %s for group %s', index, group)
                return 0
        return cls(fetch_stat, StreamingEMA(), **kwargs)

    @property
    def is_below_threshold(self):
        return self._is_below_threshold.is_set()

    def notify(self):
        """Notify the monitor that work is actively being processed"""
        self.interval.decrease()

    def wait_until_below_threshold(self):
        self._is_below_threshold.wait()

    def update_metric(self):
        update = self.fetch_stat()
        if update is None:
            # Error fetching stat. Mark as unavailable until it resolves.
            self._is_below_threshold.clear()
        else:
            self.metric.update(update)
            if self.metric.is_below_threshold(self.threshold):
                self._is_below_threshold.set()
            else:
                self._is_below_threshold.clear()
