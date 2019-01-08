from elasticsearch import Elasticsearch
from mjolnir.kafka import bulk_daemon
import pytest


def _mock_bulk_response(ok, action, status, result):
    return ok, {
        action: {
            'status': status,
            'result': result,
        }
    }


def _update_success(result, n=1):
    return [_mock_bulk_response(True, 'update', 200, result) for _ in range(n)]


def _update_missing(n=1):
    return [_mock_bulk_response(False, 'update', 404, '') for _ in range(n)]


@pytest.mark.parametrize('expected,records', [
    ({}, []),
    ({bulk_daemon.Metric.ACTION_RESULTS['updated']: 1}, _update_success('updated')),
    ({bulk_daemon.Metric.ACTION_RESULTS['created']: 2}, _update_success('created', 2)),
    ({bulk_daemon.Metric.ACTION_RESULTS['noop']: 1}, _update_success('noop')),
    ({bulk_daemon.Metric.OK_UNKNOWN: 3}, _update_success('otherthing', 3)),
    ({bulk_daemon.Metric.MISSING: 1}, _update_missing()),
    ({bulk_daemon.Metric.FAILED: 1}, [_mock_bulk_response(False, 'update', 500, '')]),
    (
        {
            bulk_daemon.Metric.ACTION_RESULTS['updated']: 4,
            bulk_daemon.Metric.ACTION_RESULTS['noop']: 2,
            bulk_daemon.Metric.MISSING: 14
        },
        _update_success('updated', 4) + _update_success('noop', 2) + _update_missing(14)
    )
])
def test_stream_to_es_stats_collection(mocker, expected, records, mock):
    mock = mocker.patch('mjolnir.kafka.bulk_daemon.streaming_bulk')
    mock.return_value = records
    for metric, _ in expected.items():
        # alhmost certainly fragile
        metric._value.set(0)
    # Dupe the records or exceptions will report empty dicts
    records = [(ok, dict(value)) for ok, value in records]
    bulk_daemon.stream_to_es(None, records)
    for metric, expected_value in expected.items():
        # almost certainly fragile
        data = metric._samples()[0][2]
        assert data == expected_value


def test_to_http_url():
    assert bulk_daemon.to_http_url('elastic1020.eqiad.wmnet:9500') == 'http://elastic1020.eqiad.wmnet:9400'
    assert bulk_daemon.to_http_url('elastic1020.eqiad.wmnet:9300') == 'http://elastic1020.eqiad.wmnet:9200'


def test_get_hosts_from_crosscluster_conf():
    conf = {
        "persistent": {
            "search": {
                "remote": {
                    "omega": {
                        "seeds": [
                            "elastic1026.eqiad.wmnet:9500",
                            "elastic1028.eqiad.wmnet:9500",
                            "elastic1029.eqiad.wmnet:9500",
                        ]
                    },
                    "psi": {
                        "seeds": [
                            "elastic1020.eqiad.wmnet:9700",
                            "elastic1023.eqiad.wmnet:9700",
                            "elastic1026.eqiad.wmnet:9700",
                        ]
                    }
                }
            }
        }
    }
    new_hosts = bulk_daemon.get_hosts_from_crosscluster_conf(conf)
    assert 'omega' in new_hosts
    assert 'psi' in new_hosts
    psi = new_hosts['psi']
    assert isinstance(psi, Elasticsearch)
    expected_hosts = [{'host': 'elastic1020.eqiad.wmnet', 'port': 9600},
                      {'host': 'elastic1023.eqiad.wmnet', 'port': 9600},
                      {'host': 'elastic1026.eqiad.wmnet', 'port': 9600}]
    assert psi.transport.hosts == expected_hosts
    assert bulk_daemon.get_hosts_from_crosscluster_conf({}) == {}
