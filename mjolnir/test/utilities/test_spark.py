from contextlib import contextmanager
import glob
import json
import mjolnir.utilities.spark
import mjolnir.utils
import os
import pytest
import shutil
import tempfile
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


@contextmanager
def tempdir():
    dir = tempfile.mkdtemp()
    try:
        yield dir
    finally:
        shutil.rmtree(dir)


@pytest.mark.parametrize('num_obs,num_features,num_workers,expected_mb', [
    (35000000, 250, 10, (24000, 26000)),
    (35000000, 50, 1, (45000, 50000)),
    (1000000, 250, 1, (6000, 8000)),
    (1000000, 50, 1, (1500, 2000)),
])
def test_autodetect_make_folds_memory_overhead(num_obs, num_features, num_workers, expected_mb):
    with tempdir() as path:
        wikis = ['testwiki']
        stats_path = os.path.join(path, '_stats.json')
        with open(stats_path, 'w') as f:
            f.write(json.dumps({
                'num_obs': {wiki: num_obs for wiki in wikis},
                'num_features': num_features,
            }))
        config = {
            'cmd_args': {
                'input': path,
                'num-workers': num_workers,
            },
            'autodetect': {
                'baseline_memory_overhead': '512M',
                'bytes_per_value': {
                    'make_folds': 29
                },
            }
        }
        overhead = mjolnir.utilities.spark.autodetect_make_folds_memory_overhead(
            config, wikis)
        assert expected_mb[0] < overhead < expected_mb[1]


@pytest.mark.parametrize('num_obs,num_features,num_workers,expected_mb', [
    (35000000, 250, 10, (15000, 20000)),
    (35000000, 50, 1, (30000, 35000)),
    (1000000, 250, 1, (5000, 6000)),
    (1000000, 50, 1, (1300, 1600)),
])
def test_autodetect_train_memory_overhead(num_obs, num_features, num_workers, expected_mb):
    with tempdir() as path:
        wikis = ['testwiki']
        stats_path = os.path.join(path, 'stats.json')
        with open(stats_path, 'w') as f:
            f.write(json.dumps({
                'wikis': {wiki: {
                    'stats': {
                        'num_observations': num_obs,
                        'features': list(range(num_features)),
                    },
                    'num_workers': num_workers
                } for wiki in wikis}
            }))
        config = {
            'cmd_args': {
                'input': path,
            },
            'autodetect': {
                'baseline_memory_overhead': '512M',
                'bytes_per_value': {
                    'train': 20
                },
            }
        }
        overhead = mjolnir.utilities.spark.autodetect_train_memory_overhead(
            config, wikis)
        assert expected_mb[0] < overhead < expected_mb[1]
