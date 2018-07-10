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
import Queue
import requests
import time


log = logging.getLogger(__name__)
REPORT_IDLE_SIGL = 'report idle'


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
        self.work_queue = Queue.Queue(10)

    def run(self):
        worker_pool = multiprocessing.dummy.Pool(self.n_workers, self._produce)
        consumer = kafka.KafkaConsumer(bootstrap_servers=self.brokers,
                                       group_id='mjolnir',
                                       enable_auto_commit=True,
                                       auto_offset_reset='latest',
                                       value_deserializer=json.loads,
                                       api_version=mjolnir.kafka.BROKER_VERSION)

        try:
            consumer.subscribe([self.topic_work])
            for record in consumer:
                if 'complete' in record.value:
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
            consumer.close()
            worker_pool.close()
            for i in range(self.n_workers):
                self.work_queue.put(None)
            worker_pool.join()

        # It is possible, if some workers have errors, for the queue to not be completely
        # emptied. Make sure it gets finished
        if self.work_queue.qsize() > 0:
            log.warning('Work queue not completely drained on shut down. Draining')
            session = requests.Session()
            while self.work_queue.qsize() > 0:
                try:
                    record = self.work_queue.get_nowait()
                    self._handle_record(session, record)
                except Exception as e:
                    log.error('Exception while shutting down daemon:')
                    log.exception(e)

        self.producer.close()

    def _reflect_end_run(self, record):
        """Reflect and end run sigil into the complete topic

        This is handled directly in the consumer thread, rather than as part of the
        work queue, to ensure that the offset is not committed to kafka until after
        processing is completed and it has been sucessfully reflected.

        Parameters
        ----------
        record : ???
            Kafka record containing the end run sigil
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
        time.sleep(10)
        record.value['offsets'] = self._get_result_offsets()

        log.info('reflecting end sigil for run %s and partition %d' %
                 (record.value['run_id'], record.value['partition']))
        future = self.ack_all_producer.send(self.topic_complete, json.dumps(record.value))
        future.add_errback(self._log_error_on_end_run)
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

    def _produce(self):
        """Run elasticsearch queries from queue and push them into topic_result"""
        session = requests.Session()
        continue_processing = True
        while continue_processing:
            record = self.work_queue.get()
            try:
                continue_processing = self._handle_record(session, record)
            except Exception:
                log.exception('Exception processing record')

    def _log_error_on_send(self, exception):
        """Log an error when a failure occurs while sending a document to the broker

        Parameters
        ----------
        exception: BaseException
            exception raised while sending a document
        """
        log.error('Failed to send a message to the broker: %s', exception)

    def _log_error_on_end_run(self, exception):
        """Log an error when a failure occurs while sending a document to the broker

        Parameters
        ----------
        exception: BaseException
            exception raised while sending a document
        """
        log.critical('Failed to send the "end run" message: %s', exception)

    def _handle_record(self, session, record):
        """Handle a single kafka record from request topic

        Parameters
        ----------
        session : requests.Session
            Session for making http requests. This must be per-thread as it is not threadsafe.
        record : ???
            Kafka record to handle

        Returns
        -------
        bool
            True if the thread should continue processing records
        """
        try:
            # Consumer finished, nothing more will arrive in the queue
            if record is None:
                return False

            # Standard execution of elasticsearch bulk query
            response = mjolnir.cirrus.make_request('msearch', session, ['http://localhost:9200'],
                                                   record.value['request'], reuse_url=True)
            future = self.producer.send(self.topic_result, json.dumps({
                'run_id': record.value['run_id'],
                'wikiid': record.value['wikiid'],
                'query': record.value['query'],
                'status_code': response.status_code,
                'text': response.text,
            }))
            future.add_errback(self._log_error_on_send)
            return True
        finally:
            self.work_queue.task_done()
