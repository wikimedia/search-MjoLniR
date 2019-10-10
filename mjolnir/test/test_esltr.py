from collections import defaultdict
from typing import Mapping

from elasticsearch import Elasticsearch
from mjolnir.esltr import LtrClient, StoredFeature, StoredFeatureSet, StoredModel
import pytest


@pytest.fixture
def feature_definition():
    return {
        'name': 'pytest_feature',
        'params': ['keywords'],
        'template_language': 'mustache',
        'template': {
            'match': {
                'title': '{{keywords}}'
            }
        }
    }


@pytest.fixture
def feature_set_definition(feature_definition: Mapping):
    return {
        'name': 'pytest_feature_set',
        'features': [feature_definition],
    }


@pytest.fixture
def model_definition(feature_set_definition: Mapping):
    return {
        'name': 'pytest_model',
        'feature_set': feature_set_definition,
        'model': {
            'type': 'magic/unicorn-dust',
            'definition': 'pixies',
        }
    }


def test_StoredFeature_round_trip(feature_definition: Mapping):
    feature = StoredFeature.from_dict(feature_definition)
    assert feature.to_dict() == feature_definition


def test_StoredFeatureSet_round_trip(feature_set_definition: Mapping):
    feature_set = StoredFeatureSet.from_dict(feature_set_definition)
    assert feature_set.to_dict() == feature_set_definition


def test_StoredModel_round_trip(model_definition: Mapping):
    model = StoredModel.from_dict(model_definition)
    assert model.to_dict() == model_definition


# Copied from elasticsearch-py test_elasticsearch/test_cases.py
class DummyTransport:
    def __init__(self, hosts, responses=None, **kwargs):
        self.hosts = hosts
        self.responses = responses
        self.call_count = 0
        self.calls = defaultdict(list)

    def perform_request(self, method, url, params=None, body=None):
        resp = 200, {}
        if self.responses:
            resp = self.responses[self.call_count]
        self.call_count += 1
        self.calls[(method, url)].append((params, body))
        return resp


class TestLtrClient:
    def setup_method(self, method):
        elastic = Elasticsearch(transport_class=DummyTransport)
        self.client = LtrClient(elastic)

    def assert_url_called(self, method, url, count=1):
        assert (method, url) in self.client.transport.calls
        calls = self.client.transport.calls[(method, url)]
        assert len(calls) == count
        return calls

    def test_create_feature_store(self):
        self.client.store.create()
        self.assert_url_called('PUT', '/_ltr')
