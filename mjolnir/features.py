"""
Integration for collecting feature vectors from elasticsearch
"""

import base64
from collections import defaultdict, namedtuple
import json
import math
import mjolnir.cirrus
import mjolnir.kafka.client
import mjolnir.spark
import os
from pyspark.accumulators import AccumulatorParam
from pyspark.ml.linalg import Vectors
from pyspark.sql import functions as F
import random
import re
import requests


def _wrap_with_page_ids(hit_page_ids, should):
    """Wrap an elasticsearch query with an ids filter.

    Parameters
    ----------
    hit_page_ids : list of ints
        Set of page ids to collect features for
    should : dict or list of dict
        Elasticsearch query for a single feature

    Returns
    -------
    string
        JSON encoded elasticsearch query
    """
    assert len(hit_page_ids) < 10000
    if not isinstance(should, list):
        should = [should]
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
                                'values': map(str, set(ids))
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
        bulk_query.append(json.dumps(self.make_search(row.hit_page_ids, row.query)))

        # elasticsearch bulk format requires each item to be on a line and the
        # request to finish with a \n
        return "%s\n" % ('\n'.join(bulk_query))


def extract_ltr_log_feature_values(response, accum):
    """Extract feature vector from ltr_log search ext
    Scan all hits and inspect the ltr_log field. Feature names are then
    lexically ordered to have consistent ordinals

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
        counts = {}
        for n, v in sorted(hit['fields']['_ltrlog'][0]['sltr_log'].iteritems()):
            features.append(v)
            counts[n] = counts.get(n, 0) + 1
        accum += counts
        yield page_id, Vectors.dense(features)


class ScriptFeature(object):
    """
    Query feature using elasticsearch script_score

    ...

    Methods
    -------
    make_query(query)
        Build the elasticsearch query
    """

    def __init__(self, name, script, lang='expression'):
        self.name = name
        self.script = script
        self.lang = lang

    def make_query(self, query):
        """Build the elasticsearch query

        Parameters
        ----------
        query : string
            User provided query term (unused)
        """
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
    """
    Query feature using elasticsearch multi_match

    ...

    Methods
    -------
    make_query(query)
        Build the elasticsearch query
    """
    def __init__(self, name, fields, minimum_should_match=1, match_type="most_fields"):
        """

        Parameters
        ----------
        name : string
            Name of the feature
        fields : list
            Fields to perform multi_match against
        minimum_should_match: int, optional
            Minimum number of fields that should match. (Default: 1)
        match_type : string, optional
            Type of match to perform. (Default: most_fields)
        """
        self.name = name
        assert len(fields) > 0
        self.fields = fields
        self.minimum_should_match = minimum_should_match
        self.match_type = match_type

    def make_query(self, query):
        """Build the elasticsearch query

        Parameters
        ----------
        query : string
            User provided query term
        """
        return {
            "multi_match": {
                "query": query,
                "minimum_should_match": self.minimum_should_match,
                "type": self.match_type,
                "fields": self.fields,
            }
        }


class DisMaxFeature(object):
    """
    Query feature using elasticsearch dis_max

    ...

    Methods
    -------
    make_query(query)
        Build the elasticsearch query
    """

    def __init__(self, name, features):
        """

        Parameters
        ----------
        name : string
            Name of the feature
        features : list
            List of queries to use with dismax
        """
        self.name = name
        assert len(features) > 0
        self.features = features

    def make_query(self, query):
        """Build the elasticsearch query

        Parameters
        ----------
        query : string
            User provided query term
        """
        return {
            "dis_max": {
                "queries": [f.make_query(query) for f in self.features]
            }
        }


class FeatureNamesAccumulator(AccumulatorParam):
    """
    Spark accumulator to keep track of the feature names used
    when retrieving feature vectors.
    Keep a dict with feature names as keys and a counter as values.
    """
    def zero(self, value):
        return dict()

    def addInPlace(self, value1, value2):
        for k, v in value2.iteritems():
            value1[k] = value1.get(k, 0) + v
        return value1


def _create_bulk_query(row, indices, feature_definitions):
    """Create an elasticsearch bulk query for provided row.

    Parameters
    ----------
    row : pyspark.sql.Row
        Row containing wikiid, query and hit_page_ids columns
    indices : dict
        Map from wikiid to elasticsearch index to query
    feature_definitions : list
        list of feature objects

    Returns
    -------
    string
    """
    bulk_query = []
    if row.wikiid in indices:
        index = indices[row.wikiid]
    else:
        # Takes advantage of aliases for the wikiid typically used by
        # CirrusSearch
        index = row.wikiid
    index_line = '{"index": "%s"}' % (index)

    for feature in feature_definitions:
        bulk_query.append(index_line)
        bulk_query.append(_wrap_with_page_ids(row.hit_page_ids,
                                              feature.make_query(row.query)))
    # elasticsearch bulk format requires each item to be on a line and the
    # request to finish with a \n
    return "%s\n" % ('\n'.join(bulk_query))


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
    features_seen = {}
    # feature ordinals (sorted by feature names)
    # ordinal_map[i] returns the feature ordinal for request i and feature_names[i]
    ordinal_map = [elts[0] for elts in sorted(enumerate(feature_names), key=lambda i:i[1])]
    ordinal_map = [elts[0] for elts in sorted(enumerate(ordinal_map), key=lambda i:i[1])]
    for i, response in enumerate(parsed['responses']):
        ordinal = ordinal_map[i]
        # These should be retried when making the queries. If we get this far then
        # no retry is possible anymore, and the default NaN value will signal to
        # throw away the hit_page_id
        if response['status'] != 200:
            continue
        features_seen[feature_names[i]] = 1
        for hit in response['hits']['hits']:
            hit_page_id = int(hit['_id'])
            features[hit_page_id][ordinal] = hit['_score']
    feature_names_accu += features_seen
    return features.items()


def enwiki_features():
    """Default set of features to collect.

    Returns
    -------
    list
    """
    return [
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
        ScriptFeature("popularity_score",
                      "pow(doc['popularity_score'].value , 0.8) / " +
                      "( pow(doc['popularity_score'].value, 0.8) + pow(8.0E-6,0.8))"),
        ScriptFeature("incoming_links",
                      "pow(doc['incoming_links'].value , 0.7) / " +
                      "( pow(doc['incoming_links'].value, 0.7) + pow(30,0.7))"),
    ]


def collect_kafka(df, brokers, feature_definitions, feature_names_accu, indices=None):
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
    feature_definitions : list
        list of feature objects defining the features to collect.
    feature_names_accu : Accumulator
        used to collect feature names
    indices : dict, optional
        map from wikiid to elasticsearch index to query. If wikiid is
        not present the wikiid will be used as index name. (Default: None)
    """
    mjolnir.spark.assert_columns(df, ['wikiid', 'query', 'hit_page_id'])
    if indices is None:
        indices = {}
    feature_names = [f.name for f in feature_definitions]
    Response = namedtuple('Response', ['status_code', 'text'])

    def kafka_handle_response(record):
        response = Response(record['status_code'], record['text'])
        for hit_page_id, features in _handle_response(response, feature_names, feature_names_accu):
            # nan features mean some sort of failure, drop the row.
            # TODO: Add some accumulator to count and report dropped rows?
            if not any(map(math.isnan, features)):
                yield [record['wikiid'], record['query'], hit_page_id, Vectors.dense(features)]

    run_id = base64.b64encode(os.urandom(16))
    offsets_start = mjolnir.kafka.client.get_offset_start(brokers)
    print 'producing queries to kafka'
    num_end_sigils = mjolnir.kafka.client.produce_queries(
        df.groupBy('wikiid', 'query').agg(F.collect_set('hit_page_id').alias('hit_page_ids')),
        brokers,
        run_id,
        lambda row: _create_bulk_query(row, indices, feature_definitions))
    print 'waiting for end run sigils'
    offsets_end = mjolnir.kafka.client.get_offset_end(brokers, run_id, num_end_sigils)
    print 'reading results from:'
    for p, (start, end) in enumerate(zip(offsets_start, offsets_end)):
        print '%d : %d to %d' % (p, start, end)
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


def collect_es(df, url_list, feature_definitions, feature_names_accu, indices=None, session_factory=requests.Session):
    """Collect feature vectors from elasticsearch

    Performs queries against a remote elasticsearch instance to collect feature
    vectors for all (query, hit_page_id) combinations in df.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Source dataframe containing wikiid, query and hit_page_id fields
        to collect feature vectors for.
    url_list : list of str
        List of URLs to send multi-search requests to. One will be chosen at
        random per partition.
    feature_definitions : list
        list of feature objects defining the features to collect.
    feature_names_accu : Accumulator used to track feature names and ordinals
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
    mjolnir.cirrus.check_idle(url_list, session_factory)

    if indices is None:
        indices = {}

    feature_names = [f.name for f in feature_definitions]

    def collect_partition(rows):
        """Generate a function that will collect feature vectors for each partition.

        Yields
        -------
        pyspark.sql.Row
            Collected feature vectors for a single (wiki, query, hit_page_id)
        """
        random.shuffle(url_list)
        url = url_list.pop()
        with session_factory() as session:
            for row in rows:
                bulk_query = _create_bulk_query(row, indices, feature_definitions)
                url, response = mjolnir.cirrus.make_request(session, url, url_list, bulk_query)
                for hit_page_id, features in _handle_response(response, feature_names, feature_names_accu):
                    # nan features mean some sort of failure, drop the row.
                    # TODO: Add some accumulator to count and report dropped rows?
                    if not any(map(math.isnan, features)):
                        yield [row.wikiid, row.query, hit_page_id, Vectors.dense(features)]

    # Protect the cluster from being overloaded. Bulk queries are issued in
    # parallel, although feature queries aren't particularly expensive. Keep
    # the total number of parallel queries < 1500
    num_executors = 1500 / len(feature_definitions)

    df_agg = (
        df
        .groupBy('wikiid', 'query')
        .agg(F.collect_set('hit_page_id').alias('hit_page_ids')))

    if df_agg.rdd.getNumPartitions() > num_executors:
        df_agg = df_agg.coalesce(num_executors)

    return (
        df_agg
        .rdd.mapPartitions(collect_partition)
        .toDF(['wikiid', 'query', 'hit_page_id', 'features']))


def _explode_ltr_model_definition(definition):
    """
    Parse a string describing a ltr featureset or model
    (featureset|model):name[@storename]

    Parameters
    ----------
    definition: string
        the model/featureset definition

    Returns
    -------
        list: 3 elements list: type, name, store
    """
    res = re.search('(featureset|model)+:([^@]+)(?:[@](.+))?$', definition)
    if res is None:
        raise ValueError("Cannot parse ltr model definition [%s]." % (definition))
    return res.groups()


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

    eltType, name, store = _explode_ltr_model_definition(model)

    def collect_partition(rows):
        """Generate a function that will collect feature vectors for each partition.

        Yields
        -------
        pyspark.sql.Row
            Collected feature vectors for a single (wiki, query, hit_page_id)
        """
        random.shuffle(url_list)
        url = url_list.pop()
        log_query = LtrLoggingQuery(eltType, name, store)
        with session_factory() as session:
            for row in rows:
                req = log_query.make_msearch(row, indices)
                url, response = mjolnir.cirrus.make_request(session, url, url_list, req)
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
    eltType, name, store = _explode_ltr_model_definition(model)
    log_query = LtrLoggingQuery(eltType, name, store)

    def kafka_handle_response(record):
        assert record['status_code'] == 200
        parsed = json.loads(record['text'])
        response = parsed['responses'][0]

        for hit_page_id, features in extract_ltr_log_feature_values(response, feature_names_accu):
            yield [record['wikiid'], record['query'], hit_page_id, features]

    run_id = base64.b64encode(os.urandom(16))
    offsets_start = mjolnir.kafka.client.get_offset_start(brokers)
    print 'producing queries to kafka'
    num_end_sigils = mjolnir.kafka.client.produce_queries(
        df.groupBy('wikiid', 'query').agg(F.collect_set('hit_page_id').alias('hit_page_ids')),
        brokers,
        run_id,
        lambda row: log_query.make_msearch(row, indices))
    print 'waiting for end run sigils'
    offsets_end = mjolnir.kafka.client.get_offset_end(brokers, run_id, num_end_sigils)
    print 'reading results from:'
    for p, (start, end) in enumerate(zip(offsets_start, offsets_end)):
        print '%d : %d to %d' % (p, start, end)
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
