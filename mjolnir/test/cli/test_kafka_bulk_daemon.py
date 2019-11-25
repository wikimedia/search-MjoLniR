from elasticsearch import Elasticsearch
from mjolnir.cli import kafka_bulk_daemon


def test_to_http_url():
    assert kafka_bulk_daemon.to_http_url('elastic1020.eqiad.wmnet:9500') == 'http://elastic1020.eqiad.wmnet:9400'
    assert kafka_bulk_daemon.to_http_url('elastic1020.eqiad.wmnet:9300') == 'http://elastic1020.eqiad.wmnet:9200'


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
    new_hosts = kafka_bulk_daemon.get_hosts_from_crosscluster_conf(conf)
    assert 'omega' in new_hosts
    assert 'psi' in new_hosts
    psi = new_hosts['psi']
    assert isinstance(psi, Elasticsearch)
    expected_hosts = [{'host': 'elastic1020.eqiad.wmnet', 'port': 9600},
                      {'host': 'elastic1023.eqiad.wmnet', 'port': 9600},
                      {'host': 'elastic1026.eqiad.wmnet', 'port': 9600}]
    assert psi.transport.hosts == expected_hosts
    assert kafka_bulk_daemon.get_hosts_from_crosscluster_conf({}) == {}
