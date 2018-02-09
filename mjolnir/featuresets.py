"""
Helpers for defining an LTR featureset
"""

from __future__ import absolute_import

import abc
import functools
import mjolnir.esltr
import re


def mangle_name(name):
    return re.sub('[^0-9a-zA-Z]', '_', name)


class Feature(object):
    __metaclass__ = abc.ABCMeta

    template_language = 'mustache'

    def __init__(self, name):
        self.name = mangle_name(name)

    @abc.abstractmethod
    def get_template(self, params):
        pass

    def make_feature_def(self, params):
        return {
            "name": self.name,
            "params": ['query_string'],
            "template_language": self.template_language,
            "template": self.get_template(params)
        }


class FieldValueFeature(Feature):
    def __init__(self, name, field=None):
        super(FieldValueFeature, self).__init__(name)
        self.field = field if field else name

    def get_template(self, params):
        return {
            'function_score': {
                'field_value_factor': {
                    'field': self.field,
                    'missing': 0,
                }
            }
        }


class MatchFeature(Feature):
    def __init__(self, name, field=None):
        super(MatchFeature, self).__init__(name)
        self.field = field if field else name

    def get_template(self, params):
        return {
            "match": {
                self.field: params['query_string'],
            }
        }


class MatchPhraseFeature(Feature):
    def __init__(self, name, field=None):
        super(MatchPhraseFeature, self).__init__(name)
        self.field = field if field else name

    def get_template(self, params):
        return {
            'match_phrase': {
                self.field: params['query_string']
            }
        }


class TokenCountRouterFeature(Feature):
    def __init__(self, nested, limit=10):
        super(TokenCountRouterFeature, self).__init__(nested.name)
        self.nested = nested
        self.limit = limit

    def get_template(self, params):
        return {
            'token_count_router': {
                'analyzer': 'text_search',
                'text': params['query_string'],
                'fallback': {'match_none': {}},
                'conditions': [
                    {
                        'gt': self.limit,
                        'query': {'match_none': {}},
                    },
                    {
                        'gt': 1,
                        'query': self.nested.get_template(params)
                    }
                ]
            }
        }


class QueryExplorerFeature(Feature):
    AGGREGATIONS = ['sum', 'min', 'max', 'mean', 'stddev']
    STATISTICS = ['raw_df', 'classic_idf', 'raw_ttf']

    def __init__(self, name, nested, aggregation, statistic):
        super(QueryExplorerFeature, self).__init__(name)
        self.nested = nested
        self.aggregation = aggregation
        self.statistic = statistic
        self.explore_type = '%s_%s' % (aggregation, statistic)
        pass

    @classmethod
    def query(cls, name_prefix, query, aggs=None):
        # Query-only features
        # Classic IDF and the total count of the term across documents
        return cls.all(name_prefix, query, aggs=aggs, stats=['classic_idf'])

    @classmethod
    def qd(cls, name_prefix, query, aggs=None):
        # Query-Document features
        # How many times the term appears in this document
        return cls.all(name_prefix, query, aggs=aggs, stats=['raw_df', 'raw_ttf'])

    @classmethod
    def all(cls, name_prefix, query, aggs=None, stats=None):
        aggs = aggs if aggs else cls.AGGREGATIONS
        stats = stats if stats else cls.STATISTICS
        features = []
        for stat in stats:
            for agg in aggs:
                feature_name = '%s_%s_%s' % (name_prefix, agg, stat)
                features.append(cls(feature_name, query, agg, stat))
        return features

    def get_template(self, params):
        return {
            "match_explorer": {
                "type": self.explore_type,
                "query": self.nested.get_template(params),
            }
        }


class TermsCountFeature(Feature):
    def __init__(self, nested):
        super(TermsCountFeature, self).__init__(nested.name)
        self.nested = nested

    def get_template(self, params):
        return {
            "match_explorer": {
                "type": "unique_terms_count",
                "query": self.nested.get_template(params)
            }
        }


class DerivedFeature(Feature):
    template_language = 'derived_expression'

    def __init__(self, name, expression):
        super(DerivedFeature, self).__init__(name)
        self.expression = expression

    def get_template(self, params):
        return self.expression


def _recursive_expression(fields, weights, pattern):
    fields = list(zip(fields, weights))
    field, weight = fields.pop()
    if weight != 1:
        field = '(%f * %s)' % (weight, field)
    expression = field
    while fields:
        field, weight = fields.pop()
        if weight != 1:
            field = '(%f * %s)' % (weight, field)
        expression = pattern % (field, expression)
    return expression


def dismax_expression(fields, tie_breaker=None, weights=None):
    if weights is None:
        weights = [1.] * len(fields)
    expression = _recursive_expression(fields, weights, 'max(%s, %s)')
    if tie_breaker is not None and len(fields) > 1:
        # = max + tie_breaker * remain
        # = max + tie_breaker * (sum - max)
        # = max + tie_breaker * sum - tie_breaker * max
        # = (1 - tie_breaker) * max + tie_breaker * sum
        sum_expression = _recursive_expression(fields, weights, '%s + %s')
        expression = '((1 - %f) * (%s)) + (%f * (%s))' % (tie_breaker, expression, tie_breaker, sum_expression)
    return expression


def field_features(name, plain=True, explorer=True):
    if explorer == 'partial':
        explorer = functools.partial(QueryExplorerFeature.query, aggs=['sum'])
    elif explorer:
        explorer = QueryExplorerFeature.all
    else:
        def explorer(x, y):
            return []

    features = []
    match = MatchFeature('%s_match' % (name), name)
    features.append(match)
    features += explorer(name, match)

    if plain:
        match_plain = MatchFeature('%s_plain_match' % (name), name + '_plain')
        features.append(match_plain)
        # Simlute a weighted dis-max with no tie breaker instead of recalculating
        features.append(DerivedFeature(
            '%s_dismax_plain' % (name),
            # TODO: the 3x here is completely arbitrary to prefer plain
            dismax_expression([match.name, match_plain.name], weights=[3, 1])))
        features += explorer(name + '_plain', match_plain)

    return features


def enwiki_features():
    """Default set of features to collect.

    Returns
    -------
    list
    """
    primary_fields = ['title', 'redirect.title', 'heading', 'opening_text', 'text', 'category', 'auxiliary_text']
    secondary_fields = []

    features = []
    for field in primary_fields:
        features += field_features(field)
    for field in secondary_fields:
        features += field_features(field, explorer='partial')

    # Suggest doesn't have a plain field
    # TODO: Does query explorer make sense with suggest?
    features += field_features('suggest', plain=False, explorer=False)
    # all_near_match is also special
    features.append(MatchFeature('all_near_match'))

    # Some fake dismax's to avoid recalculating
    features.append(DerivedFeature(
        'redirect_or_suggest_dismax',
        dismax_expression(['redirect_title_dismax_plain', 'suggest_match'], tie_breaker=0.1)))
    features.append(DerivedFeature(
        'text_or_opening_text_dismax',
        dismax_expression(['text_dismax_plain', 'opening_text_dismax_plain'], tie_breaker=0.1)))

    features.append(TokenCountRouterFeature(
        MatchPhraseFeature('all_phrase_match', 'all')))
    features.append(TokenCountRouterFeature(
        MatchPhraseFeature('all_plain_phrase_match', 'all.plain')))
    features.append(DerivedFeature(
        'all_phrase_match_dismax_plain',
        # TODO: Weights arbitrarily came from cirrussearch
        dismax_expression(['all_phrase_match', 'all_plain_phrase_match'], weights=[0.6, 1])))

    # query independent features

    # Doesn't seem worthwhile to do this per field, should be basicaly
    # the same everywhere.
    name_terms_count = 'title_unique_terms'
    features.append(TermsCountFeature(MatchFeature(name_terms_count, 'title')))
    # The different analysis chains and this being a unique count means
    # the two are not necessarily the same
    name_plain_unique_terms = 'title_plain_unique_terms'
    features.append(TermsCountFeature(MatchFeature(name_plain_unique_terms, 'title.plain')))
    # Difference between term counts.
    features.append(DerivedFeature(
        'title_unique_terms_diff_plain',
        '%s - %s' % (name_terms_count, name_plain_unique_terms)))

    features.append(FieldValueFeature('popularity_score'))
    features.append(FieldValueFeature('incoming_links'))
    features.append(FieldValueFeature('text.word_count'))

    return features


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

    params = {'query_string': '{{query_string}}'}
    built = [f.make_feature_def(params) for f in features]
    feature_set.add_features(built)
    return feature_set
