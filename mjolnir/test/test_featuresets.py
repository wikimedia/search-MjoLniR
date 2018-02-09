from __future__ import absolute_import
import mjolnir.featuresets

import json
import os
import pytest


def test_create_enwiki(fixtures_dir):
    """Do nothing test asserts happy path doesnt blow up"""
    features = mjolnir.featuresets.enwiki_features()
    assert len(features) > 0
    params = {'query_string': '{{query_string}}'}
    definitions = [f.make_feature_def(params) for f in sorted(features, key=lambda f: f.name)]

    fixture = os.path.join(fixtures_dir, 'featuresets.json')
    dumped = json.dumps(definitions, indent=4, sort_keys=True)
    if os.path.exists(fixture):
        with open(fixture, 'r') as f:
            data = json.load(f)
        definitions = json.loads(dumped)
        assert len(data) == len(definitions)
        for i, (a, b) in enumerate(zip(data, definitions)):
            assert a == b, 'Feature %i' % (i)
    else:
        with open(fixture, 'w') as f:
            f.write(dumped)


def test_field_value_feature():
    f = mjolnir.featuresets.FieldValueFeature('foo')
    template = f.get_template(None)
    assert f.name == 'foo'
    assert template['function_score']['field_value_factor']['field'] == 'foo'

    f = mjolnir.featuresets.FieldValueFeature('foo', 'bar')
    template = f.get_template(None)
    assert f.name == 'foo'
    assert template['function_score']['field_value_factor']['field'] == 'bar'


def test_match_feature():
    f = mjolnir.featuresets.MatchFeature('foo')
    template = f.get_template({'query_string': '{{query_string}}'})
    assert f.name == 'foo'
    assert '{{query_string}}' == template['match']['foo']

    f = mjolnir.featuresets.MatchFeature('foo', 'bar')
    template = f.get_template({'query_string': '{{query_string}}'})
    assert f.name == 'foo'
    assert '{{query_string}}' == template['match']['bar']

    f = mjolnir.featuresets.MatchFeature('redirect.title')
    template = f.get_template({'query_string': '{{query_string}}'})
    assert f.name == 'redirect_title'
    assert 'redirect.title' in template['match']

    f = mjolnir.featuresets.MatchFeature('redirect.title_plain', 'redirect.title.plain')
    template = f.get_template({'query_string': '{{query_string}}'})
    assert f.name == 'redirect_title_plain'
    assert 'redirect.title.plain' in template['match']


def test_match_phrase_feature():
    f = mjolnir.featuresets.MatchPhraseFeature('foo')
    template = f.get_template({'query_string': '{{query_string}}'})
    assert f.name == 'foo'
    assert '{{query_string}}' == template['match_phrase']['foo']

    f = mjolnir.featuresets.MatchPhraseFeature('foo', 'bar')
    template = f.get_template({'query_string': '{{query_string}}'})
    assert f.name == 'foo'
    assert '{{query_string}}' == template['match_phrase']['bar']


def test_token_count_router_feature():
    nested = mjolnir.featuresets.MatchFeature('foo')
    f = mjolnir.featuresets.TokenCountRouterFeature(nested)
    template = f.get_template({'query_string': '{{query_string}}'})
    nested_template = nested.get_template({'query_string': '{{query_string}}'})
    assert template['token_count_router']['text'] == '{{query_string}}'
    assert template['token_count_router']['conditions'][1]['query'] == nested_template


def test_query_explorer_feature():
    nested = mjolnir.featuresets.MatchFeature('foo')
    f = mjolnir.featuresets.QueryExplorerFeature('foo', nested, 'sum', 'raw_df')
    template = f.get_template({'query_string': '{{query_string}}'})
    nested_template = nested.get_template({'query_string': '{{query_string}}'})
    assert template['match_explorer']['type'] == 'sum_raw_df'
    assert template['match_explorer']['query'] == nested_template


def test_derived_feature():
    f = mjolnir.featuresets.DerivedFeature('foo', 'max(bar, baz)')
    feat_def = f.make_feature_def(None)
    assert feat_def['template_language'] == 'derived_expression'
    assert feat_def['name'] == 'foo'
    assert feat_def['template'] == 'max(bar, baz)'


def test_field_features():
    feats = mjolnir.featuresets.field_features('foo', plain=False, explorer=False)
    assert len(feats) == 1
    assert feats[0].name == 'foo_match'

    feats = mjolnir.featuresets.field_features('foo.bar', plain=True, explorer=False)
    assert set([f.name for f in feats]) == set(['foo_bar_match', 'foo_bar_plain_match', 'foo_bar_dismax_plain'])

    feats = mjolnir.featuresets.field_features('foo', plain=False, explorer='partial')
    assert set([f.name for f in feats]) == set([
        'foo_match', 'foo_sum_classic_idf'])

    feats = mjolnir.featuresets.field_features('foo', plain=True, explorer='partial')
    assert len(feats) == 5

    feats = mjolnir.featuresets.field_features('foo', plain=False, explorer=True)
    assert len(feats) == 16

    feats = mjolnir.featuresets.field_features('bar', plain=True, explorer=True)
    assert len(feats) == 33


@pytest.mark.parametrize("fields,tie_breaker,weights,expected", [
    # Plain application
    (['a', 'b'], None, None, 'max(a, b)'),
    (['a', 'b', 'c'], None, None, 'max(a, max(b, c))'),
    # With tie breaker
    (['a', 'b'], 0.1, None, '((1 - 0.100000) * (max(a, b))) + (0.100000 * (a + b))'),
    (['a', 'b', 'c'], 0.2, None, '((1 - 0.200000) * (max(a, max(b, c)))) + (0.200000 * (a + b + c))'),
    # With weights
    (['a', 'b'], None, [1, 1], 'max(a, b)'),
    (['a', 'b'], None, [0.3, 1], 'max((0.300000 * a), b)'),
    (['a', 'b'], None, [0.3, 0.3], 'max((0.300000 * a), (0.300000 * b))'),
    (['a', 'b', 'c'], None, [1, 3, 1], 'max(a, max((3.000000 * b), c))'),
    # Tie breaker and weights
    (['a', 'b', 'c'], 0.25, [0.1, 0.2, 0.3], '((1 - 0.250000) * (max((0.100000 * a), max((0.200000 * b), (0.300000 * c))))) + (0.250000 * ((0.100000 * a) + (0.200000 * b) + (0.300000 * c)))'),  # noqa: E501
])
def test_dismax_expression(fields, tie_breaker, weights, expected):
    expression = mjolnir.featuresets.dismax_expression(fields, tie_breaker, weights)
    assert expression == expected
