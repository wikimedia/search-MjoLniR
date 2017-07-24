"""
Client for pushing elasticsearch queries into a kafka topic and reading results
back from a second kafka topic. This runs on the analytics side as part of feature
collection.
"""

import json
import mjolnir.spark
import mjolnir.kafka
from pyspark.streaming.kafka import KafkaUtils, OffsetRange
import kafka
import kafka.common


def _make_producer(brokers):
    return kafka.KafkaProducer(bootstrap_servers=brokers,
                               compression_type='gzip')


def produce_queries(df, brokers, run_id, create_bulk_query, topic=mjolnir.kafka.TOPIC_REQUEST):
    """Push feature collection queries into kafka.

    Write out the feature requests as elasticsearch multi-search queries to kafka.
    These will be picked up by the daemon on relforge, and the collected back in
    the collect_results function.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Source data frame to collect feature vectors for
    brokers : list of str
        List of kafka brokers used to bootstrap access into the kafka cluster.
    run_id : str
        A unique identifier for this data collection run
    create_bulk_query : callable
        Function accepting a row from df that returns a string
        containing an elasticsearch multi-search query.
    topic : str, optional
        The topic to produce queries to

    Returns
    -------
    int
        The number of end run sigils to wait for in the complete topic.
    """

    def produce_partition(rows):
        producer = _make_producer(brokers)
        for row in rows:
            producer.send(topic, json.dumps({
                'run_id': run_id,
                'request': create_bulk_query(row),
                'wikiid': row.wikiid,
                'query': row.query,
            }))
        producer.close()

    mjolnir.spark.assert_columns(df, ['wikiid', 'query', 'hit_page_ids'])
    df.rdd.foreachPartition(produce_partition)

    # Send a sigil value to indicate this run is complete. The consumer will copy this
    # into TOPIC_COMPLETE so we know it's done.
    producer = _make_producer(brokers)
    partitions = producer.partitions_for(topic)
    for p in partitions:
        producer.send(topic, partition=p, value=json.dumps({
            'run_id': run_id,
            'complete': True,
            'partition': p
        }))
    producer.close()
    return len(partitions)


def get_offset_start(brokers, topic=mjolnir.kafka.TOPIC_RESULT):
    """Find the current ending offset for all partitions in topic.

    By calling this prior to producing requests we know all responses come
    after these offsets.
    TODO: This naming doesn't feel right...

    Parameters
    ----------
    brokers : list of str
    topic : str

    Returns
    -------
    list of int
    """
    consumer = kafka.KafkaConsumer(bootstrap_servers=brokers)
    parts = consumer.partitions_for_topic(topic)
    if parts is None:
        return None
    partitions = [kafka.TopicPartition(topic, p) for p in parts]
    consumer.assign(partitions)
    return [consumer.position(p) for p in partitions]


def get_offset_end(brokers, run_id, num_end_sigils, topic=mjolnir.kafka.TOPIC_COMPLETE):
    """ Find the offset of the last message of our run

    The 'end run' message gets reflected, by the client running on relforge,
    back into TOPIC_COMPLETE into all partitions. This reads those partitions
    and looks for the ending offset of all partitions based on that reflected
    message

    Parameters
    ----------
    brokers : list of str
    run_id : str
        Unique identifier for this run
    num_end_sigils : int
        The number of unique end run sigils to expect. This should be the number of partitions
        of the topic requests were produced to.
    topic : str, optional
        Topic to look for end run messages in

    Returns
    -------
    list of ints
        The offset of the end run message for all partitions
    """
    consumer = kafka.KafkaConsumer(bootstrap_servers=brokers,
                                   # The topic we are reading from is very low volume,
                                   # containing only reflected end run sigils. To make
                                   # sure we don't miss one start at the beginning.
                                   auto_offset_reset='earliest',
                                   value_deserializer=json.loads)
    parts = consumer.partitions_for_topic(topic=mjolnir.kafka.TOPIC_COMPLETE)
    if parts is None:
        raise RuntimeError("topic %s missing" % topic)

    partitions = [kafka.TopicPartition(topic, p) for p in consumer.partitions_for_topic(topic)]
    consumer.assign(partitions)
    # Tracks the maximum reported offset in the response topic
    offsets_end = [-1] * num_end_sigils
    # Tracks the sigils that have been seen for the request topics
    # Uses a set incase duplicate messages are sent somehow, to ensure
    # we see a message for all expected partitions
    seen_sigils = set()
    for message in consumer:
        if 'run_id' in message.value and message.value['run_id'] == run_id and 'complete' in message.value:
            print 'found sigil for run %s and partition %d' % (message.value['run_id'], message.value['partition'])
            for partition, offset in enumerate(message.value['offsets']):
                offsets_end[partition] = max(offsets_end[partition], offset)
            seen_sigils.add(message.value['partition'])
            # Keep reading until all sigils have been reflected.
            if len(seen_sigils) >= num_end_sigils:
                consumer.close()
                return offsets_end
    consumer.close()
    raise RuntimeError("Finished consuming, but %d partitions remain" % (len(partitions) - len(seen_sigils)))


def collect_results(sc, brokers, receive_record, offsets_start, offsets_end, run_id):
    """
    Parameters
    ----------
    sc : pyspark.SparkContext
    brokers : list of str
    receive_record : callable
        Callable receiving a json decoded record from kafka. It must return
        either an empty list on error, or a 3 item tuple containing
        hit_page_id as int, query as str, and features as DenseVector
    offsets_start : list of int
        Per-partition offsets to start reading at
    offsets_end : list of int
        Per-partition offsets to end reading at
    run_id : str
        unique identifier for this run

    Returns
    -------
    pyspark.RDD
        RDD containing results of receive_record
    """

    offset_ranges = []
    if offsets_start is None:
        offsets_start = get_offset_start(brokers, mjolnir.kafka.TOPIC_RESULT)

    if offsets_start is None:
        raise RuntimeError("Cannot fetch offset_start, topic %s should have been created" % mjolnir.kafka.TOPIC_RESULT)
    for partition, (start, end) in enumerate(zip(offsets_start, offsets_end)):
        offset_ranges.append(OffsetRange(mjolnir.kafka.TOPIC_RESULT, partition, start, end))
    assert not isinstance(brokers, basestring)
    kafka_params = {"metadata.broker.list": ','.join(brokers)}
    # If this ends up being too much data from kafka, blowing up memory in the
    # spark executors, we could chunk the offsets and union together multiple RDD's.
    return (
        KafkaUtils.createRDD(sc, kafka_params, offset_ranges)
        .map(lambda (k, v): json.loads(v))
        .filter(lambda rec: 'run_id' in rec and rec['run_id'] == run_id)
        .flatMap(receive_record))
