"""
Daemon for collecting elasticsearch queries from a kafka topic, running them,
and pushing the results into a second kafka topic. This runs on the production
side of the network to have access to relforge servers.
"""

from __future__ import absolute_import
import json
import kafka
import kafka.common
import kafka.consumer.subscription_state
import logging
import multiprocessing.dummy
import mjolnir.cirrus
import mjolnir.kafka
import queue
import requests
import time


log = logging.getLogger(__name__)


def iter_queue(queue):
    """Yield items from a queue"""
    while True:
        record = queue.get()
        # Queue is finished, nothing more will arrive
        if record is None:
            return
        yield record


class Daemon(object):
    def __init__(self, brokers, n_workers=5, topic_work=mjolnir.kafka.TOPIC_REQUEST,
                 topic_result=mjolnir.kafka.TOPIC_RESULT, topic_complete=mjolnir.kafka.TOPIC_COMPLETE,
                 max_request_size=4*1024*1024):
        self.brokers = brokers
        self.n_workers = n_workers
        self.topic_work = topic_work
        self.topic_result = topic_result
        self.topic_complete = topic_complete
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
        # TODO: 10 items? No clue how many is appropriate...10 seems reasonable
        # enough.  We want enough to keep the workers busy, but not so many
        # that the commited offsets are siginficantly ahead of the work
        # actually being performed.
        self.work_queue = queue.Queue(10)

    def run(self):
        try:
            self.consume(self.iter_records())
        finally:
            self.producer.close()
            self.ack_all_producer.close()

    def iter_records(self):
        consumer = kafka.KafkaConsumer(bootstrap_servers=self.brokers,
                                       group_id='mjolnir',
                                       enable_auto_commit=True,
                                       auto_offset_reset='latest',
                                       value_deserializer=lambda x: json.loads(x.decode('utf8')),
                                       api_version=mjolnir.kafka.BROKER_VERSION)

        consumer.subscribe([self.topic_work])
        try:
            while True:
                poll_response = consumer.poll(timeout_ms=60000)
                if not poll_response:
                    continue
                offsets = {}
                for tp, records in poll_response.items():
                    for record in records:
                        yield record.value
                    offsets[tp] = kafka.OffsetAndMetadata(records[-1].offset + 1, '')
                # Wait for all the work to complete
                self.work_queue.join()
                consumer.commit_async(offsets)
        finally:
            consumer.close()

    def consume(self, records):
        def work_fn():
            self._handle_records(iter_queue(self.work_queue))

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

    # Time to wait before reflecting a sigil to hope that everything
    # has completed.
    REFLECT_WAIT = 10

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
        log.info('received end run sigil. Waiting for queue to drain')
        self.work_queue.join()

        log.info('work drained. Waiting around to make sure everything really finishes')
        # Flush is perhaps not strictly necessary, the producer sends almost immediately, but
        # lets be explicit here.
        self.producer.flush()
        # Also give time for the messages to make it around the kafka cluster. No clue what
        # an appropriate time is here. This can't be too long, or the consumer group coordinator
        # will kick us out of the group and this end run sigil will get re-processed.
        # This is equal to the default broker replica.lag.time.max.ms, which controls how
        # much a replica can fall behind the master.
        # TODO: The kafka protocol docs suggest offset requests always go to the leader,
        # this might be unnecessary.
        time.sleep(self.REFLECT_WAIT)
        record['offsets'] = self._get_result_offsets()

        log.info('reflecting end sigil for run %s and partition %d' %
                 (record['run_id'], record['partition']))
        future = self.ack_all_producer.send(self.topic_complete, json.dumps(record).encode('utf8'))
        future.add_errback(lambda e: log.critical(
            'Failed to send the "end run" message: %s', e))
        # TODO: Is this enough to guarantee delivery? Not sure what failures cases are.
        future.get()

    def _get_result_offsets(self):
        """Get the latest offsets for all partitions in topic"""
        consumer = kafka.KafkaConsumer(bootstrap_servers=self.brokers,
                                       auto_offset_reset='latest',
                                       api_version=mjolnir.kafka.BROKER_VERSION)
        partitions = [kafka.TopicPartition(self.topic_result, p)
                      for p in consumer.partitions_for_topic(self.topic_result)]
        consumer.assign(partitions)
        consumer.seek_to_end()
        offsets = [consumer.position(tp) for tp in partitions]
        consumer.close()
        return offsets

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
                response = mjolnir.cirrus.make_request('msearch', session, ['http://localhost:9200'],
                                                       record['request'], reuse_url=True)
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
