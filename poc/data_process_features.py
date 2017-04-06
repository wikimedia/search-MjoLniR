import json
import os
import subprocess
import base64

from pyspark.streaming.kafka import KafkaUtils, OffsetRange
import kafka
import kafka.common

import config
from utils import spark_utils

KAFKA_BOOTSTRAP = 'kafka1012.eqiad.wmnet:9092'
KAFKA_WORK_LOG = 'relforge_queries'
KAFKA_RESULT_LOG = 'relforge_results'

RELFORGE_INDEX='{"index": "crosswiki_enwiki_content"}'

def wrap_with_page_ids(hit_page_ids, should):
    if type(should) is not list:
        should = [should]
    if len(hit_page_ids) >= 9999:
        raise ValueError("Too many page ids: %d" % (len(hit_page_ids)))
    return json.dumps({
        "_source": False,
        "from": 0,
        "size": 9999,
        "query": {
            "bool": {
                "filter": {
                    'ids': {
                        'values': map(str, set(hit_page_ids)),
                    }
                },
                "should": should,
                "disable_coord": True,
            }
        }
    })

class ScriptFeature(object):
    def __init__(self, name, script, lang='expression'):
        self.name = name
        self.script = script
        self.lang = lang

    def make_query(self, query):
        return {
            "function_score": {
                "score_mode": "sum",
                "boost_mode": "sum",
                "functions": [
                    {
                        "script_score": {
                            "script": {
                                "inline": self.script,
                                "lang": self.lang,
                            }
                        }
                    }
                ]
            }
        }


class MultiMatchFeature(object):
    def __init__(self, name, fields, minimum_should_match=1, match_type="most_fields"):
        self.name = name
        assert len(fields) > 0
        self.fields = fields
        self.minimum_should_match = minimum_should_match
        self.match_type = match_type

    def make_query(self, query):
        return {
            "multi_match": {
                "query": query,
                "minimum_should_match": self.minimum_should_match,
                "type": self.match_type,
                "fields": self.fields,
            }
        }

class DisMaxFeature(object):
    def __init__(self, name, features):
        self.name = name
        assert len(features) > 0
        self.features = features

    def make_query(self, query):
        return {
            "dis_max": {
                "queries": [f.make_query(query) for f in self.features]
            }
        }

def gen_produce_partition(features, run_id):
    def f(rows):
        producer = kafka.KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP)
        i = 0
        for row in rows:
            bulk_query = []
            try:
                for feature in features:
                    bulk_query.append(RELFORGE_INDEX)
                    bulk_query.append(wrap_with_page_ids(row.hit_page_ids, feature.make_query(row.query)))
            except ValueError:
                # TODO How to send error logs back?
                continue
            producer.send(KAFKA_WORK_LOG, json.dumps({
                'run_id': run_id,
                'req_id': row.query,
                'request': "\n".join(bulk_query)
            }))
            i += 1
            if i % 100 == 0:
                producer.flush()
    return f


def produce_queries(hive, features, run_id):
    """
    Dump out the click data as es queries, to be sent to relforge
    and the feature logs collected
    """

    query = """
        SELECT query, COLLECT_SET(hit_page_id) as hit_page_ids
        FROM click_data
        GROUP BY query
    """
    hive.read.parquet(config.CLICK_DATA).registerTempTable('click_data')
    hive.sql(query).foreachPartition(gen_produce_partition(features, run_id))
    # Send a sigil value to indicate this run is complete. The consumer will copy this
    # into KAFKA_RESULT_LOG so we know it's done.
    # TODO: Certainly only works with a single partition and consumer. Do we also have
    # potential timing issues?
    producer = kafka.KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP)
    future = producer.send(KAFKA_WORK_LOG, json.dumps({
        'run_id': run_id,
        'complete': True,
    }))
    record = future.get()
    print("Sendt end sigil at offset %d" % (record.offset))



def get_offset_start():
    # boldly assume single partition, for now
    kafkaClient = kafka.SimpleClient([KAFKA_BOOTSTRAP])
    # -1 means 'the offset of the next coming message'
    reqs = [kafka.common.OffsetRequestPayload(KAFKA_RESULT_LOG, 0, -1, 1)]
    resps = kafkaClient.send_offset_request(reqs)
    offsets = {}
    for resp in resps:
        return resp.offsets[0]
    return None


def get_offset_end(offset_start, run_id):
    # boldly assume single partition, for now
    # Start up a consumer to wait for our completion sigil
    consumer = kafka.KafkaConsumer(bootstrap_servers=[KAFKA_BOOTSTRAP],
                                   auto_offset_reset='latest',
                                   value_deserializer=json.loads)
    tp = kafka.TopicPartition(KAFKA_RESULT_LOG, 0)
    consumer.assign([tp])
    consumer.seek(tp, offset_start)
    for message in consumer:
        if 'run_id' not in message.value:
            pass
        if message.value['run_id'] == run_id and 'complete' in message.value:
            return message.offset
    return None


def gen_receive(features, run_id):
    expected_count = len(features)
    # Receives messages sent back from relforge
    def f(parsed):
        if not parsed['run_id'] == run_id:
            # Some other process working at same time as us
            return []
        if not 'status_code' in parsed:
            # Some other kind of message, perhaps end sigil?
            return []
        if not parsed['status_code'] == 200:
            # TODO: error handling
            return []
        if not 'responses' in parsed:
            # TODO: error handling
            return []
        query = parsed['req_id']
        # Parse result of elasticsearch into a feature vector
        features = {}
        for response in parsed['responses']:
            # TODO: Check response['status'] ? if not 200 then what?
            for hit in response['hits']['hits']:
                page_id = hit['_id']
                if page_id not in features:
                    features[page_id] = []
                features[page_id].append(hit['_score'])
        for page_id in features.keys():
            assert len(features[page_id]) == expected_count
        return [[int(page_id), query] + v for (page_id, v) in features.iteritems()]
    return f


def collect_results(sc, offset_start, offset_end, features, run_id):
    offset_ranges = [OffsetRange(KAFKA_RESULT_LOG, 0, offset_start, offset_end)]
    kafka_params = {"metadata.broker.list": KAFKA_BOOTSTRAP}
    col_names = ['hit_page_id', 'query'] + ['feature_%s' % (f.name) for f in features]
    return (KafkaUtils.createRDD(sc, kafka_params, offset_ranges)
        .map(lambda (k, v): json.loads(v))
        .flatMap(gen_receive(features, run_id))
        .toDF(col_names)
    )


def main():
    sc, hive = spark_utils._init("LTR: process")

    # Build the set of features we will collect vectors for
    features = [
        MultiMatchFeature('title', ["title.plain^1", "title^3"]),
        MultiMatchFeature('category', ["category.plain^1", "category^3"]),
        MultiMatchFeature('heading', ["heading.plain^1", "heading^3"]),
        MultiMatchFeature('auxiliary_text', ["auxiliary_text.plain^1", "auxiliary_text^3"]),
        MultiMatchFeature('file_text', ["file_text.plain^1", "file_text^3"]),
        DisMaxFeature('redirect_or_suggest_dismax', [
            MultiMatchFeature(None, ["redirect.title.plain^1", "redirect.title^3"]),
            MultiMatchFeature(None, ["suggest"]),
        ]),
        DisMaxFeature('text_or_opening_text_dismax', [
            MultiMatchFeature(None, ["text.plain^1", "text^3"]),
            MultiMatchFeature(None, ["opening_text.plain^1", "opening_text^3"]),
        ]),
        MultiMatchFeature('all_near_match', ["all_near_match^2"]),
        ScriptFeature("popularity_score", "pow(doc['popularity_score'].value , 0.8) / ( pow(doc['popularity_score'].value, 0.8) + pow(8.0E-6,0.8))"),
        ScriptFeature("incoming_links", "pow(doc['incoming_links'].value , 0.7) / ( pow(doc['incoming_links'].value, 0.7) + pow(30,0.7))"),
    ]

    # Generate a random ID for this run
    run_id = base64.b64encode(os.urandom(16))
    print("Starting up feature collection with run_id: %s" % (run_id))

    offset_start = get_offset_start()
    if offset_start is None:
        print("Failed to find starting offset ?!?!?")
        return
    print("Start offset for results: %d" % (offset_start))

    # start up the producer to send all our records to the kafka work log
    print("Producing queries to kafka")
    produce_queries(hive, features, run_id)
    print("Finished producing queries, waiting for completion")

    # start up a consumer to detect the end offset
    offset_end = get_offset_end(offset_start, run_id)
    if offset_end is None:
        print("Failed to find ending offset ?!?!?")
        return
    print("End offset for results: %d" % (offset_end))
    print("Collecting results")
    df = collect_results(sc, offset_start, offset_end, features, run_id)
    print("Writing out collected features to hdfs")
    df.write.parquet(config.FEATURE_LOGS)



if __name__ == "__main__":
    main()

