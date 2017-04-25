"""
Integration for collecting feature vectors from elasticsearch
"""

import json
import math
import mjolnir.spark
from pyspark.ml.linalg import Vectors
from pyspark.sql import functions as F
import random
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


def _make_request(session, url, url_list, bulk_query, num_retries=5):
    failures = 0
    while True:
        try:
            return url, session.get(url, data=bulk_query)
        except requests.ConnectionError as e:
            failures += 1
            if failures >= num_retries or len(url_list) == 0:
                raise e
            # TODO: This is only desirable if url_list is a list of actual
            # servers. If the url_list is a loadbalancer like LVS then we
            # want to keep using the existing url.
            url = url_list.pop()


def _handle_response(response, num_features, hit_page_ids):
    """Parse an elasticsearch response from requests into a feature vector.

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

    features = {hit_page_id: [float('nan')] * num_features for hit_page_id in hit_page_ids}
    for i, response in enumerate(parsed['responses']):
        # Again, retries? something else?
        assert response['status'] == 200
        for hit in response['hits']['hits']:
            page_id = int(hit['_id'])
            features[page_id][i] = hit['_score']
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


def collect(df, url_list, feature_definitions, indices=None, session_factory=requests.Session):
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

    def collect_partition(rows):
        """Generate a function that will collect feature vectors for each partition.

        Yields
        -------
        pyspark.sql.Row
            Collected feature vectors for a single (wiki, query, hit_page_id)
        """
        num_features = len(feature_definitions)
        random.shuffle(url_list)
        url = url_list.pop()
        with session_factory() as session:
            for row in rows:
                bulk_query = _create_bulk_query(row, indices, feature_definitions)
                url, response = _make_request(session, url, url_list, bulk_query)
                for hit_page_id, features in _handle_response(response, num_features, row.hit_page_ids):
                    # nan features mean some sort of failure, drop the row.
                    # TODO: Add some accumulator to count and report dropped rows?
                    if not any(map(math.isnan, features)):
                        yield [row.wikiid, row.query, hit_page_id, Vectors.dense(features)]

    return (
        df
        .groupBy('wikiid', 'query')
        .agg(F.collect_set('hit_page_id').alias('hit_page_ids'))
        .rdd.mapPartitions(collect_partition)
        .toDF(['wikiid', 'query', 'hit_page_id', 'features'])
        .withColumn('features', mjolnir.spark.add_meta(df._sc, F.col('features'), {
            'features': [f.name for f in feature_definitions]
        })))
