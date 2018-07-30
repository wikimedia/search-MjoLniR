import glob
import mjolnir.utilities.spark
import os
import pytest
import yaml


@pytest.mark.parametrize("a,b,expected", [
    ({}, {}, {}),
    ({'a': 1}, {}, {'a': 1}),
    ({}, {'b': 2}, {'b': 2}),
    ({'a': 2}, {'a': 3}, {'a': 3}),
    ({'a': 1, 'b': 2}, {'b': 3}, {'a': 1, 'b': 3}),
])
def test_dict_merge(a, b, expected):
    assert mjolnir.utilities.spark.dict_merge(a, b) == expected


@pytest.mark.parametrize("template_vars,environment,expected", [
    # Most basic run through
    ({}, {}, {}),
    # Variables can reference each other
    ({'a': 'foo%(b)s', 'b': 'bar'}, {}, {'a': 'foobar', 'b': 'bar'}),
    # Variables can reference each other from different sources
    ({'a': 'foo%(b)s'}, {'b': 'bar'}, {'a': 'foobar', 'b': 'bar'}),
    # template vars take precedence over environment (and vars are cast to strings)
    ({'a': 1}, {'a': 2}, {'a': '1'}),
])
def test_build_template_vars(template_vars, environment, expected):
    res = mjolnir.utilities.spark.build_template_vars(template_vars, environment, 'MARKER')
    # Not testing this for simplicity
    del res['marker']
    assert res == expected


def sort_fixture(d):
    if not isinstance(d, dict):
        return d
    return {k: sort_fixture(v) for k, v in sorted(d.items(), key=lambda x: x[0])}


def generate_fixtures(test_name):
    # Load fixtures for test_load_config, it's too verbose to put inline
    tests = []
    fixtures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fixtures')
    for test_file in glob.glob(os.path.join(fixtures_dir, test_name, '*.test')):
        expect_file = os.path.splitext(test_file)[0] + '.expect'
        tests.append((test_file, expect_file))
    return ('test_file,expect_file', tests)


def compare_fixture(expect_file, generated):
    as_yaml = yaml.dump(sort_fixture(generated), default_flow_style=False)
    if os.path.isfile(expect_file):
        with open(expect_file, 'r') as f:
            current = sort_fixture(yaml.safe_load(as_yaml))
            fixture = sort_fixture(yaml.safe_load(f))
            assert current == fixture
    else:
        with open(expect_file, 'w') as f:
            f.write(as_yaml)


@pytest.mark.parametrize(*generate_fixtures('load_config'))
def test_load_config(monkeypatch, test_file, expect_file):
    monkeypatch.setenv('HOME', '/home/pytest')
    monkeypatch.setenv('USER', 'pytest')

    with open(test_file, 'r') as f:
        working_dir, global_config, profiles = mjolnir.utilities.spark.load_config(f, 'marker', {
            'mjolnir_dir': '/srv/mjolnir',
        })
    compare_fixture(expect_file, {
        'global_config': global_config,
        'profiles': profiles
    })


@pytest.mark.parametrize(*generate_fixtures('build_spark_command'))
def test_build_spark_command(test_file, expect_file):
    with open(test_file, 'r') as f:
        test = yaml.safe_load(f)
    cmd = mjolnir.utilities.spark.build_spark_command(test)
    compare_fixture(expect_file, cmd)


@pytest.mark.parametrize(*generate_fixtures('build_mjolnir_utility'))
def test_build_mjolnir_utility(test_file, expect_file):
    with open(test_file, 'r') as f:
        test = yaml.safe_load(f)
    cmd = mjolnir.utilities.spark.build_mjolnir_utility(test)
    compare_fixture(expect_file, cmd)
