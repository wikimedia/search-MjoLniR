"""
Integration for collecting feature vectors from elasticsearch ltr plugin
"""

from __future__ import absolute_import
import base64
from collections import defaultdict, OrderedDict
import json
import mjolnir.cirrus
import mjolnir.kafka.client
import mjolnir.spark
import mjolnir.utils
import os
from pyspark.accumulators import AccumulatorParam
from pyspark.ml.linalg import Vectors
from pyspark.sql import functions as F
import random
import requests


class LtrLoggingQuery(object):
    """Model or featureset stored in a featurestore

    ...

    Methods
    -------
    make_query(query)
        Build the elasticsearch query
    """

    def __init__(self, elementType, name, store=None, queryParam='query_string'):
        """Prepare a sltr query (requires the ltr plugin)

        Parameters
        ----------
        name : string
            Name of the model or featureset to log
        elementType: string
            Either set or model
        queryParam:
            name of the param to pass the query string
        store:
            name of the feature store
        """
        self.name = name
        self.store = store
        self.elementType = elementType
        self.queryParam = queryParam

    def make_search(self, ids, query):
        """Build the elasticsearch query

        Parameters
        ----------
        ids : list of ints
            Set of page ids to collect features for
        query : string
            User provided query term (unused)

        Returns
        -------
        An elasticsearch search request object
        """

        query = {
            "_source": False,
            "from": 0,
            "size": 9999,
            "stats": ["mjolnir"],
            "query": {
                "bool": {
                    "filter": [
                        # wrap inside the filter so we bypass score computation during the query phase
                        # feature scores will be computed only once during the fetch phase
                        {
                            "sltr": {
                                "_name": "sltr_log",
                                "params": {
                                    self.queryParam: query
                                },
                                self.elementType: self.name
                            }
                        },
                        {
                            'ids': {
                                'values': sorted(map(str, set(ids)))
                            }
                        }
                    ]
                }
            },
            "ext": {
                "ltr_log": {
                    "log_specs": [
                        {
                            "named_query": "sltr_log",
                            "missing_as_zero": True,
                        }
                    ]
                }
            }
        }
        return query

    def make_msearch(self, row, indices):
        """Build the elasticsearch query

        Parameters
        ----------
        query : string
            User provided query term (unused)
        row : pyspark.sql.Row
            Row containing wikiid, query and hit_page_ids columns
        indices : dict
            Map from wikiid to elasticsearch index to query

        Returns
        -------
        string
            An elasticsearch msearch request encoded as a json string
        """

        # some duplicated code for creating a single search, this is mostly to
        # reuse msearch request handling in cirrus.py. This possibly could be
        # used to send multiple queries once.
        bulk_query = []
        if row.wikiid in indices:
            index = indices[row.wikiid]
        else:
            # Takes advantage of aliases for the wikiid typically used by
            # CirrusSearch
            index = row.wikiid
        bulk_query.append('{"index": "%s"}' % (index))
        bulk_query.append(json.dumps(self.make_search(row.hit_page_ids, row.query), sort_keys=True))

        # elasticsearch bulk format requires each item to be on a line and the
        # request to finish with a \n
        return "%s\n" % ('\n'.join(bulk_query))


def extract_ltr_log_feature_values(response, accum):
    """Extract feature vector from ltr_log search ext
    Scan all hits and inspect the ltr_log field.

    Parameters
    ----------
    response : dict
        elasticsearch search response
    accum : Accumulator
        spark accumulator to collect feature names

    Yields
    ------
    pyspark.sql.Row
        Collected feature vectors for a single (wiki, query, hit_page_id)
    """

    for hit in response['hits']['hits']:
        page_id = int(hit['_id'])
        features = []
        counts = OrderedDict()
        for v in hit['fields']['_ltrlog'][0]['sltr_log']:
            score = v['value']
            n = v['name']
            features.append(score)
            counts[n] = counts.get(n, 0) + 1
        accum += counts
        yield page_id, Vectors.dense(features)


class FeatureNamesAccumulator(AccumulatorParam):
    """
    Spark accumulator to keep track of the feature names used
    when retrieving feature vectors.
    Keep a dict with feature names as keys and a counter as values.
    """
    def zero(self, value):
        return OrderedDict()

    def addInPlace(self, value1, value2):
        for k, v in value2.items():
            value1[k] = value1.get(k, 0) + v
        return value1


def _handle_response(response, feature_names, feature_names_accu):
    """Parse an elasticsearch response from requests into a feature vector.

    Parameters
    ----------
    response : dict
        msearch responses
    feature_names : list
        feature names
    feature_names_accu : Accumulator
        count logged features

    Returns
    ------
    list of tuples
        List contains two item tuples, each with hit_page_id first
        and a list of features collected second.
    """
    # TODO: retries? something else?
    assert response.status_code == 200
    parsed = json.loads(response.text)
    assert 'responses' in parsed, response.text

    features = defaultdict(lambda: [float('nan')] * len(feature_names))
    features_seen = OrderedDict()
    for i, response in enumerate(parsed['responses']):
        # These should be retried when making the queries. If we get this far then
        # no retry is possible anymore, and the default NaN value will signal to
        # throw away the hit_page_id
        if response['status'] != 200:
            features_seen[feature_names[i]] = 0
            continue
        features_seen[feature_names[i]] = 1
        for hit in response['hits']['hits']:
            hit_page_id = int(hit['_id'])
            features[hit_page_id][i] = hit['_score']
    feature_names_accu += features_seen
    return features.items()


def collect_from_ltr_plugin(df, url_list, model, feature_names_accu, indices=None, session_factory=requests.Session):
    """Collect feature vectors from elasticsearch and the ltr plugin

    Performs queries against a remote elasticsearch instance to collect feature
    vectors using the features defined in a named featureset or model for all
    (query, hit_page_id) combinations in df.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Source dataframe containing wikiid, query and hit_page_id fields
        to collect feature vectors for.
    url_list : list of str
        List of URLs to send multi-search requests to. One will be chosen at
        random per partition.
    model : string
        definition of the model/featureset: "featureset:name", "model:name" or "featureset:name@storeName"
    indices : dict, optional
        map from wikiid to elasticsearch index to query. If wikiid is
        not present the wikiid will be used as index name. (Default: None)
    session_factory : callable, optional
        Used to create new sessions. (Default: requests.Session)

    Returns
    -------
    pyspark.sql.DataFrame
        Collected feature vectors with one row per unique (wikiid, query, hit_page_id). All
        feature columns are prefixed with feature_.
    """

    mjolnir.spark.assert_columns(df, ['wikiid', 'query', 'hit_page_id'])

    if indices is None:
        indices = {}

    eltType, name, store = mjolnir.utils.explode_ltr_model_definition(model)

    def collect_partition(rows):
        """Generate a function that will collect feature vectors for each partition.

        Yields
        -------
        pyspark.sql.Row
            Collected feature vectors for a single (wiki, query, hit_page_id)
        """
        # mjolnir.cirrus.make_request will modify the url list. Take a copy to ensure
        # the modifications don't escape.
        partition_url_list = list(url_list)
        random.shuffle(partition_url_list)
        log_query = LtrLoggingQuery(eltType, name, store)
        with session_factory() as session:
            for row in rows:
                req = log_query.make_msearch(row, indices)
                response = mjolnir.cirrus.make_request('msearch', session, partition_url_list, req)
                assert response.status_code == 200
                parsed = json.loads(response.text)
                assert 'responses' in parsed, response.text
                resp = parsed['responses'][0]

                for hit_page_id, features in extract_ltr_log_feature_values(resp, feature_names_accu):
                    yield [row['wikiid'], row['query'], hit_page_id, features]

    return (
        df
        .groupBy('wikiid', 'query')
        .agg(F.collect_set('hit_page_id').alias('hit_page_ids'))
        .rdd.mapPartitions(collect_partition)
        .toDF(['wikiid', 'query', 'hit_page_id', 'features']))


def collect_from_ltr_plugin_and_kafka(df, brokers, model, feature_names_accu, indices=None):
    """Collect feature vectors from elasticsearch via kafka

    Pushes queries into a kafka topic and retrieves results from a second kafka topic.
    A daemon must be running on relforge to collect the queries and produce results.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Source dataframe containing wikiid, query and hit_page_id fields
        to collect feature vectors for.
    brokers : list of str
        List of kafka brokers used to bootstrap access into the kafka cluster.
    model : string
        definition of the model/featureset: "featureset:name", "model:name" or "featureset:name@storeName"
    feature_names_accu : Accumulator
        used to collect feature names
    indices : dict, optional
        map from wikiid to elasticsearch index to query. If wikiid is
        not present the wikiid will be used as index name. (Default: None)
    """
    mjolnir.spark.assert_columns(df, ['wikiid', 'query', 'hit_page_id'])
    if indices is None:
        indices = {}
    eltType, name, store = mjolnir.utils.explode_ltr_model_definition(model)
    log_query = LtrLoggingQuery(eltType, name, store)

    def kafka_handle_response(record):
        assert record['status_code'] == 200
        parsed = json.loads(record['text'])
        response = parsed['responses'][0]
        meta = record['meta']

        for hit_page_id, features in extract_ltr_log_feature_values(response, feature_names_accu):
            yield [meta['wikiid'], meta['query'], hit_page_id, features]

    run_id = base64.b64encode(os.urandom(16)).decode('ascii')
    offsets_start = mjolnir.kafka.client.get_offset_start(brokers)
    print('producing queries to kafka')
    num_end_sigils = mjolnir.kafka.client.produce_queries(
        df.groupBy('wikiid', 'query').agg(F.collect_set('hit_page_id').alias('hit_page_ids')),
        brokers, run_id,
        create_es_query=lambda row: log_query.make_msearch(row, indices),
        meta_keys=['wikiid', 'query'])
    print('waiting for end run sigils')
    offsets_end = mjolnir.kafka.client.get_offset_end(brokers, run_id, num_end_sigils)
    print('reading results from:')
    for p, (start, end) in enumerate(zip(offsets_start, offsets_end)):
        print('%d : %d to %d' % (p, start, end))
    return (
        mjolnir.kafka.client.collect_results(
            df._sc,
            brokers,
            kafka_handle_response,
            offsets_start,
            offsets_end,
            run_id)
        .toDF(['wikiid', 'query', 'hit_page_id', 'features'])
        # We could have gotten duplicate data from kafka. Clean them up.
        .drop_duplicates(['wikiid', 'query', 'hit_page_id']))


def collect(df, model, url_list=None, brokers=None, indices=None, session_factory=requests.Session):
    """Collect feature values from elasticsearch

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input data. Must have wikiid, query and hit_page_id columns
    model : str
        Feature set definition. (featureset|model):name[@storename]
    url_list : list of str, optional
        List of http urls to elasticsearch servers to query directly
        from executors.
    brokers : list of str, optional
        List of kafka broker strings to use instead of a direct connection
        for feature collection.
    indices : map str -> str, optional
        Map from wikiid to index to collect features from.
    session_factory : requests.Session

    Returns
    -------
    df_features : pyspark.sql.DataFrame
        Dataframe with wikiid, query, hit_page_id and features columns.
        The new features column is a pyspark.ml.linalg.Vector
    feature_names_accu : pyspark.Accumulator
        Collects ordered dict of feature names to usage counts. Can be
        used to ensure all the features were collected the same number
        of times, as expected.
    """
    if brokers and url_list:
        raise ValueError('cannot specify brokers and url_list')
    feature_names_accu = df._sc.accumulator(OrderedDict(), FeatureNamesAccumulator())
    if brokers:
        df_res = collect_from_ltr_plugin_and_kafka(df, brokers, model, feature_names_accu, indices)
    elif url_list:
        df_res = collect_from_ltr_plugin(df, url_list, model, feature_names_accu, indices, session_factory)
    else:
        raise ValueError('Unknown collection method')

    return df_res, feature_names_accu
