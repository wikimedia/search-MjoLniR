import mjolnir.utilities.upload
from mjolnir.featuresets import DerivedFeature, MatchFeature
import pytest


@pytest.mark.parametrize("definitions,features,expected", [
    ([], [], []),
    ([MatchFeature('foo')], ['foo'], ['foo']),
    ([MatchFeature('foo'), MatchFeature('bar')], ['foo'], ['foo']),
    ([MatchFeature('foo'), MatchFeature('bar'), DerivedFeature('foobar', 'max(foo, bar)')],
        ['foobar'], ['foo', 'bar', 'foobar'])
])
def test_feature_dependencies(definitions, features, expected):
    definitions = [x.make_feature_def({"query_string": ""}) for x in definitions]
    selected = mjolnir.utilities.upload.make_minimal_feature_set(definitions, features)
    selected_names = [f['name'] for f in selected]
    assert selected_names == expected
