"""
Helpers for defining an LTR featureset
"""

from __future__ import absolute_import

import mjolnir.esltr


def make_feature_def(query, params):
    return {
        'name': query.name,
        'params': params.keys(),
        'template_language': 'mustache',
        'template': query.make_query(params['query_string'])
    }


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


def create_feature_set(name, features, store_name=None, index=None):
    ltr = mjolnir.esltr.Ltr()
    feature_store = ltr.get_feature_store(store_name)
    if not feature_store.exists():
        feature_store.create()
    feature_set = feature_store.get_feature_set(name)
    if feature_set.exists():
        raise Exception("Feature set already exists")
    if index:
        feature_set.validation = {
            "index": index,
            "params": {
                "query_string": 'yabba dabba doo',
            },
        }

    params = {'query_stirng': '{{query_string}}'}
    built = [make_feature_def(f, params) for f in features]
    feature_set.add_features(built)
    return feature_set
