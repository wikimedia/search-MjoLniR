"""
Client for pushing elasticsearch queries into a kafka topic and reading results
back from a second kafka topic. This runs on the analytics side as part of feature
collection.
"""

import base64
from collections import namedtuple
import json
import os
import time

import kafka

import mjolnir.spark
import mjolnir.kafka

# Kafka client configuration
ClientConfig = namedtuple('ClientConfig', [
    'brokers', 'req_topic', 'resp_topic', 'control_topic'])
# 4 fields and 3 defaults means brokers is still required
ClientConfig.__new__.__defaults__ = (
    mjolnir.kafka.TOPIC_REQUEST,
    mjolnir.kafka.TOPIC_RESULT,
    mjolnir.kafka.TOPIC_COMPLETE)

# A specific range of a kafka.TopicPartition
OffsetRange = namedtuple('OffsetRange', ['tp', 'start', 'end'])


def _make_producer(client_config):
    return kafka.KafkaProducer(bootstrap_servers=client_config.brokers,
                               compression_type='gzip')


def ratelimit(rows, rate, clock=time.monotonic):
    """Apply per-second rate limit to iterable.

    Parameters
    ----------
    rows : iterable
    rate : int
        Number of rows to allow through per second
    clock : callable
        0-arity function returning current time in seconds
    """
    next_reset = clock() + 1
    num_rows = 0
    for row in rows:
        time_remaining = next_reset - clock()
        if time_remaining < 0 or num_rows >= rate:
            if time_remaining > 0:
                time.sleep(time_remaining)
            num_rows = 0
            next_reset = clock() + 1
        num_rows += 1
        yield row


def produce_queries(
        df, client_config, run_id, create_es_query, meta_keys,
        max_concurrent_producer=20, rate_limit_per_sec=1500
):
    """Push msearch queries into kafka.

    Write out the dataframe rows to kafka as elasticsearch multi-search queries.
    These will be picked up by the msearch daemon, and the collected back in
    the collect_results function.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Source data frame to collect feature vectors for
    client_config : ClientConfig
    run_id : str
        A unique identifier for this data collection run
    create_es_query : callable
        Function accepting a row from df that returns a string
        containing an elasticsearch query.
    meta_keys : list of str
        List of Row fields to include in the message metadata. These
        will be returned when consuming responses.
    max_concurrent_producer : int
        Maximum number of concurrent producers to use. This helps keep
        unnecessary load off of the kafka clusters.
    rate_limit_per_sec : int
        Maximum number of records to produce per second to kafka
        across all producers

    Returns
    -------
    int
        The number of end run sigils to wait for in the complete topic.
    """

    meta_keys = set(meta_keys)
    # If we have less than max_concurrent_producer partitions this
    # wont change anything.
    rdd = df.rdd.coalesce(max_concurrent_producer)
    # No guarantee spark will run all of the partitions concurrently,
    # but set the per-partition rate limit assuming all of them are.
    partition_rate_limit = rate_limit_per_sec / rdd.getNumPartitions()

    def produce_partition(rows):
        producer = _make_producer(client_config)
        for row in ratelimit(rows, partition_rate_limit):
            producer.send(client_config.req_topic, json.dumps({
                'run_id': run_id,
                'request': create_es_query(row),
                'meta': {k: v for k, v in row.asDict().items() if k in meta_keys},
            }).encode('utf8'))
        producer.close()

    mjolnir.spark.assert_columns(df, meta_keys)
    rdd.foreachPartition(produce_partition)

    # Send a sigil value to indicate this run is complete. The consumer will copy this
    # into TOPIC_COMPLETE so we know it's done.
    producer = _make_producer(client_config)
    partitions = producer.partitions_for(client_config.req_topic)
    for p in partitions:
        producer.send(client_config.req_topic, partition=p, value=json.dumps({
            'run_id': run_id,
            'complete': True,
            'partition': p
        }).encode('utf8'))
    producer.close()
    return len(partitions)


def offsets_for_times(consumer, partitions, timestamp):
    """Augment KafkaConsumer.offsets_for_times to not return None

    Parameters
    ----------
    consumer : kafka.KafkaConsumer
        This consumer must only be used for collecting metadata, and not
        consuming. API's will be used that invalidate consuming.
    partitions : list of kafka.TopicPartition
    timestamp : number
        Timestamp, in seconds since unix epoch, to return offsets for.

    Returns
    -------
    dict from kafka.TopicPartition to integer offset
    """
    # Kafka uses millisecond timestamps
    timestamp_ms = int(timestamp * 1000)
    response = consumer.offsets_for_times({p: timestamp_ms for p in partitions})
    offsets = {}
    for tp, offset_and_timestamp in response.items():
        if offset_and_timestamp is None:
            # No messages exist after timestamp. Fetch latest offset.
            consumer.assign([tp])
            consumer.seek_to_end(tp)
            offsets[tp] = consumer.position(tp)
        else:
            offsets[tp] = offset_and_timestamp.offset
    return offsets


def offset_range_for_timestamp_range(brokers, start, end, topic):
    """Determine OffsetRange for a given timestamp range

    Parameters
    ----------
    client_config : ClientConfig
    start : number
        Unix timestamp in seconds
    end : number
        Unix timestamp in seconds
    topic : str
        Topic to fetch offsets for

    Returns
    -------
    list of OffsetRange or None
        Per-partition ranges of offsets to read
    """
    consumer = kafka.KafkaConsumer(bootstrap_servers=brokers)
    partitions = consumer.partitions_for_topic(topic)
    if partitions is None:
        # Topic does not exist.
        return None
    partitions = [kafka.TopicPartition(topic, p) for p in partitions]
    o_start = offsets_for_times(consumer, partitions, start)
    o_end = offsets_for_times(consumer, partitions, end)
    return [OffsetRange(tp, o_start[tp], o_end[tp]) for tp in partitions]


def wait_for_sigils(client_config, run_id, num_end_sigils):
    """Wait for the end run sigils to be reflected

    The 'end run' message gets reflected, by the client running the msearch
    daemon, back into TOPIC_COMPLETE into all partitions. This waits until
    all sigils that were sent have been reflected, indicating everything sent
    before the sigil has been processed and is available in the result topic.

    Parameters
    ----------
    client_config : ClientConfig
    run_id : str
        Unique identifier for this run
    num_end_sigils : int
        The number of unique end run sigils to expect. This should be the number of partitions
        of the topic requests were produced to.
    """
    consumer = kafka.KafkaConsumer(bootstrap_servers=client_config.brokers,
                                   # The topic we are reading from is very low volume,
                                   # containing only reflected end run sigils. To make
                                   # sure we don't miss one start at the beginning.
                                   auto_offset_reset='earliest',
                                   value_deserializer=lambda x: json.loads(x.decode('utf8')))
    parts = consumer.partitions_for_topic(client_config.control_topic)
    if parts is None:
        raise RuntimeError("topic %s missing" % client_config.control_topic)

    # Tracks the sigils that have been seen for the request topics
    # Uses a set incase duplicate messages are sent somehow, to ensure
    # we see a message for all expected partitions
    seen_sigils = set()
    consumer.subscribe([client_config.control_topic])
    try:
        for message in consumer:
            if 'run_id' in message.value and message.value['run_id'] == run_id and 'complete' in message.value:
                print('found sigil for run %s and partition %d' % (message.value['run_id'], message.value['partition']))
                seen_sigils.add(message.value['partition'])
                # Keep reading until all sigils have been reflected.
                if len(seen_sigils) >= num_end_sigils:
                    return
        raise RuntimeError("Finished consuming, but %d partitions remain" % (num_end_sigils - len(seen_sigils)))
    finally:
        consumer.close()


def kafka_to_rdd(sc, client_config, offset_ranges):
    """Read ranges of kafka partitions into an RDD.

    Parameters
    ----------
    sc : pyspark.SparkContext
    client_config : ClientConfig
    offset_ranges : list of OffsetRange
        List of topic partitions along with ranges to read. Start
        and end of range are inclusive.

    Returns
    -------
    pyspark.RDD
        Contents of the specified offset_ranges
    """
    def read_offset_range(offset_range):
        if offset_range.end <= offset_range.start:
            # Raise exception?
            return
        # After serialization round trip these fail an isinstance check.
        # re-instantiate so we have the expected thing.
        tp = kafka.TopicPartition(*offset_range.tp)
        consumer = kafka.KafkaConsumer(bootstrap_servers=client_config.brokers,
                                       value_deserializer=lambda x: json.loads(x.decode('utf8')))
        try:
            consumer.assign([tp])
            consumer.seek(tp, offset_range.start)
            while True:
                poll_response = consumer.poll(timeout_ms=10000)
                if poll_response and tp in poll_response:
                    for message in poll_response[tp]:
                        if message.offset > offset_range.end:
                            break
                        yield message.value
                if consumer.position(tp) >= offset_range.end:
                    break
        finally:
            consumer.close()

    return (
        # TODO: This isn't the same as assigning each offset_range to a separate
        # partition, but it doesn't seem like pyspark allows us to do that. Often
        # enough this seems to achieve the same thing, but without guarantees.
        sc.parallelize(offset_ranges, len(offset_ranges))
        .flatMap(read_offset_range)
    )


def collect_results(sc, client_config, receive_record, start, end, run_id):
    """
    Parameters
    ----------
    sc : pyspark.SparkContext
    client_config : ClientConfig
    receive_record : callable
        Callable receiving a json decoded record from kafka. It should return
        a list and the resulting rdd will have a record per result returned.
    start : int
        Timestamp, in seconds since unix epoch, at which to start looking for records
    end : int
        Timestamp at which to stop looking for records.
    run_id : str
        unique identifier for this run

    Returns
    -------
    pyspark.RDD
        RDD containing results of receive_record
    """
    # Decide what offsets we need.
    offset_ranges = offset_range_for_timestamp_range(
            client_config.brokers, start, end, client_config.resp_topic)
    if offset_ranges is None:
        raise RuntimeError('Could not retrieve offset ranges for result topic. Does it exist?')

    # If this ends up being too much data from kafka, blowing up memory in the
    # spark executors, we could chunk the offsets and union together multiple RDD's.
    return (
        kafka_to_rdd(sc, client_config, offset_ranges)
        .filter(lambda rec: 'run_id' in rec and rec['run_id'] == run_id)
        .flatMap(receive_record))


def msearch(df, client_config, meta_keys, create_es_query, handle_response):
    """Run an msearch against each row of the input dataframe

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    client_config : ClientConfig
        Configuration of brokers and topics to communicate with
    meta_keys : list of str
        List of fields in df to pass through to handle_response
    create_es_query : callable
        Transform row from df into a valid elasticsearch bulk request. This
        must be a str ready for POST exactly following the msearch spec.
    handle_response : callable
        Processes individual responses from elasticesarch. A single dict argument
        is provided with the keys `status_code`, `text` and `meta`. Status code is
        the http status code. Text contains the raw text result. Meta is a dict
        containing the key/value pairs from meta_keys.

    Returns
    -------
    pyspark.RDD
        The result of running msearch on the input. The shape is determined
        by the results of the `handle_response` argument
    """
    # assumes client_config was a list of brokers to connect to
    # and the topics take defaults
    if not isinstance(client_config, ClientConfig):
        client_config = ClientConfig(client_config)

    run_id = base64.b64encode(os.urandom(16)).decode('ascii')
    # Adjust the start/end times by one minute to give a little flexibility in data arriving.
    start = time.time() - 60
    num_end_sigils = produce_queries(df, client_config, run_id, create_es_query, meta_keys)
    wait_for_sigils(client_config, run_id, num_end_sigils)
    end = time.time() + 60
    return collect_results(df._sc, client_config, handle_response, start, end, run_id)
