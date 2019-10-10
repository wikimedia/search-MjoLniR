"""Interface for elasticesarch ltr plugin management api

Doesn't try too hard, simply wraps some url/method handling so we dont have to
duplicate it elsewhere. All responses are return as-is from the management API.
Responses from elasticsearch containing an 'error' key are converted to
exceptions.

FIXME: template parameters are awkward
"""

from collections import OrderedDict
import re
from typing import cast, Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, TypeVar, Union

from elasticsearch import Elasticsearch
from elasticsearch.client.utils import AddonClient, NamespacedClient, query_params, _make_path

from mjolnir.utils import explode_ltr_model_definition


_T = TypeVar('_T')


class FeatureStoreClient(NamespacedClient):
    """CRUD operations for feature store indices"""
    @query_params()
    def create(self, store: Optional[str] = None, params: Optional[Mapping] = None) -> Mapping:
        return self.transport.perform_request('PUT', _make_path('_ltr', store), params=params)

    @query_params()
    def delete(self, store: Optional[str] = None, params: Optional[Mapping] = None) -> Mapping:
        return self.transport.perform_request('DELETE', _make_path('_ltr', store), params=params)

    @query_params()
    def exists(self, store: Optional[str] = None, params: Optional[Mapping] = None) -> bool:
        if params is None:
            params = {}
        response = self.search(**params)
        if store is None:
            store = '_default_'
        return 'stores' in response and store in response['stores']

    @query_params()
    def search(self, params: Optional[Mapping] = None) -> Mapping:
        """List available feature stores"""
        return self.transport.perform_request('GET', _make_path('_ltr'), params=params)


class CrudClient(NamespacedClient):
    """CRUD operations for data stored in the ltr plugin.

    Base class for objects stored in the ltr plugin. Should be subclassed
    per use case."""
    def __init__(self, client: Elasticsearch, store_type: str):
        super().__init__(client)
        self.store_type = '_' + store_type

    def _req(
        self, method: str, store: Optional[str], suffix: Iterable[str],
        body: Optional[Mapping], params: Optional[Mapping]
    ) -> Mapping:
        path = _make_path('_ltr', store, self.store_type, *suffix)
        return self.transport.perform_request(method, path, body=body, params=params)

    @query_params()
    def create(
        self, name: str, body: Mapping,
        store: Optional[str] = None, params: Optional[Mapping] = None
    ) -> Mapping:
        """Create a new named object.

        Parameters
        ----------
        name:
            The name of the object to create
        body:
            Object definition
        store:
            The name of the feature store to scope operation to. When
            None the default feature store is selected.
        """
        return self._req('PUT', store, [name], body, params)

    @query_params()
    def update(
        self, name: str, body: Mapping,
        store: Optional[str] = None, params: Optional[Mapping] = None
    ) -> Mapping:
        """Update a named object.

        Parameters
        ----------
        name:
            The name of the object to update
        body:
            Object definition
        store:
            The name of the feature store to scope operation to. When
            None the default feature store is selected.
        """
        return self._req('POST', store, [name], body, params)

    @query_params('routing')
    def delete(self, name: str, store: Optional[str] = None, params: Optional[Mapping] = None) -> Mapping:
        """Delete a named object.

        Parameters
        ----------
        name:
            The name of the object to delete
        store:
            The name of the feature store to scope operation to. When
            None the default feature store is selected.
        routing: str
            Specific routing value

        Throws
        ------
        elasticsearch.exceptions.NotFoundError:
            when the object does not exist.
        """
        return self._req('DELETE', store, [name], None, params)

    @query_params('routing')
    def get(
        self, name: str, store: Optional[str] = None,
        params: Optional[Mapping] = None
    ) -> Mapping:
        """Get a named object.

        Parameters
        ----------
        name:
            The name of the object to fetch
        store:
            The name of the feature store to scope operation to. When
            None the default feature store is selected.
        routing: str
            Specific routing value

        Throws
        ------
        elasticsearch.exceptions.NotFoundError:
            when the object does not exist.
        """
        return self._req('GET', store, [name], None, params)

    @query_params('routing')
    def exists(self, name: str, store: Optional[str] = None, params: Optional[Mapping] = None) -> bool:
        """Check existence of named object.

        Parameters
        ----------
        name:
            The name of the object to fetch
        store:
            The name of the feature store to scope operation to. When
            None the default feature store is selected.
        routing: str
            Specific routing value
        """
        response = self._req('HEAD', store, [name], None, params)
        # The return type signature for _req is a lie, the value depends on the
        # request. HEAD requests returns bool.
        return response  # type: ignore

    @query_params('prefix', 'from', 'size')
    def search(self, store: Optional[str] = None, params: Optional[Mapping] = None) -> Mapping:
        """List objects stored in the ltr plugin

        Parameters
        ----------
        store:
            The name of the feature store to scope operation to. When
            None the default feature store is selected.
        prefix: str
            Prefix of stored named to search for
        from: int
            Offset to start results at
        size: int
            Maximum number of returned results
        """
        return self._req('GET', store, [], None, params)


class CacheClient(NamespacedClient):
    @query_params()
    def clear(self, store: Optional[str] = None, params: Optional[Mapping] = None) -> Mapping:
        """Clear the ltr model cache

        Parameters
        ----------
        store:
            The name of the feature store to scope operation to. When
            None the default feature store is selected.
        """
        path = _make_path('_ltr', store, '_clearcache')
        return self.transport.perform_request('POST', path, params=params)

    @query_params()
    def stats(self, params: Optional[Mapping] = None) -> Mapping:
        """Query the ltr model cache stats api

        The cache stats api retieves cluster-wide stats about the ltr model cache.
        """
        path = _make_path('_ltr', '_cachestats')
        return self.transport.perform_request('GET', path)


class FeatureClient(CrudClient):
    """Manage individually stored features"""
    def __init__(self, client: Elasticsearch):
        super().__init__(client, 'feature')


class FeatureSetClient(CrudClient):
    """Manage sets of stored features"""
    def __init__(self, client: Elasticsearch):
        super().__init__(client, 'featureset')

    # TODO: self._req('POST', store, [feature_set_name, '_createmodel'], body, params)
    # TODO: self._req('POST', store, [feature_set_name, '_addfeatures', feature_prefix_query], params=params)

    @query_params('merge')
    def add_features(
        self,
        feature_set_name: str,
        body: Mapping,
        store: Optional[str] = None,
        params: Optional[Mapping] = None
    ) -> Mapping:
        """Add feature definitions to existing feature set.

        Parameters
        ----------
        feature_set_name:
            The name of the feature set to scope operation to
        body:
            Feature definitions
        store:
            The name of the feature store to scope operation to. When
            None the default feature store is selected.
        merge: bool
            If true, update set by merging features, feature with
            same names are updated new features are appended.
        """
        return self._req('POST', store, [feature_set_name, '_addfeatures'], body, params)


class ModelClient(CrudClient):
    """Manage stored ranking models"""
    def __init__(self, client: Elasticsearch):
        super().__init__(client, 'model')


class LtrClient(AddonClient):
    """Elasticsearch-learning-to-rank low-level client.

    The instance has attributes ``cache``, ``store``, ``feature``,
    ``feature_set``, and ``model`` that provide access to instances of
    :class:`~mjolnir.esltr.CacheClient`,
    :class:`~mjolnir.esltr.FeatureStoreClient`,
    :class:`~mjolnir.esltr.FeatureClient`,
    :class:`~mjolnir.esltr.FeatureSetClient` and
    :class:`~mjolnir.esltr.ModelClient` respectively.

    As a low-level client the operations only throw exceptions on
    non-2xx responses. All 2xx responses from the plugin return the
    decoded json response.
    """
    namespace = 'ltr'

    def __init__(self, client: Elasticsearch):
        super().__init__(client)
        self.cache = CacheClient(client)
        self.store = FeatureStoreClient(client)
        self.feature = FeatureClient(client)
        self.feature_set = FeatureSetClient(client)
        self.model = ModelClient(client)


# Domain objects stored in the plugin. These offer a very simple interface for
# constructing requests and interpreting results of objects stored in the ltr
# plugin.
#
# Note that when encoding these objects to send to the plugin they are almost always
# wrapped in a single-value dict containing the type. So for example to add a feature
# to a feature store:
#
#  feature = StoredFeature('test', ['keywords'], 'mustache', {"match":{"title":"{{keywords}}"}})
#  response = ltr_client.feature.create(feature.name, {'feature': feature.to_dict()})

class StoredFeature:
    """A single named ltr feature"""
    def __init__(
        self, name: str, params: Sequence[str],
        template_language: str, template: Union[str, Mapping]
    ) -> None:
        self.name = name
        self.params = params
        self.template_language = template_language
        self.template = template

    @classmethod
    def from_dict(cls, response: Mapping) -> 'StoredFeature':
        return cls(**response)

    @classmethod
    def get(cls, client: LtrClient, name: str, store: Optional[str] = None) -> 'StoredFeature':
        response = client.feature.get(name, store)
        return cls.from_dict(response['_source']['feature'])

    def create(self, client: LtrClient, store: Optional[str] = None) -> Mapping:
        return client.feature.create(self.name, {'feature': self.to_dict()}, store)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            'name': self.name,
            'params': list(self.params),
            'template_language': self.template_language,
            'template': self.template
        }


class StoredFeatureSet:
    """A named set of ltr features"""
    def __init__(self, name: str, features: Sequence[StoredFeature]) -> None:
        self.name = name
        self.features = features

    @classmethod
    def from_dict(cls, response: Mapping) -> 'StoredFeatureSet':
        defs = [StoredFeature.from_dict(child) for child in response['features']]
        return cls(response['name'], defs)

    @classmethod
    def get(cls, client: LtrClient, name: str, store: Optional[str] = None) -> 'StoredFeatureSet':
        response = client.feature_set.get(name, store)
        return cls.from_dict(response['_source']['featureset'])

    def create(self, client: LtrClient, store: Optional[str] = None) -> Mapping:
        return client.feature_set.create(self.name, {'featureset': self.to_dict()}, store)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            'name': self.name,
            'features': [feature.to_dict() for feature in self.features]
        }

    @property
    def feature_names(self) -> Sequence[str]:
        return [f.name for f in self.features]

    def __len__(self) -> int:
        return len(self.features)


class ValidationRequest:
    def __init__(self, index: str, params: Mapping[str, Any]) -> None:
        self.index = index
        self.params = params

    def to_dict(self):
        return {
            'index': self.index,
            'params': self.params
        }


class StoredModel:
    """A stored ranking model"""
    def __init__(
        self, name: str, feature_set: StoredFeatureSet,
        type: str, definition: Union[str, Mapping]
    ) -> None:
        self.name = name
        self.feature_set = feature_set
        self.type = type
        self.definition = definition

    @classmethod
    def from_dict(cls, response: Mapping) -> 'StoredModel':
        return cls(
            response['name'],
            StoredFeatureSet.from_dict(response['feature_set']),
            response['model']['type'],
            response['model']['definition'])

    @classmethod
    def get(cls, client: LtrClient, name: str, store: Optional[str] = None) -> 'StoredModel':
        response = client.model.get(name, store)
        return cls.from_dict(response['_source']['model'])

    def create(
        self, client: LtrClient,
        validation: Optional[ValidationRequest] = None,
        store: Optional[str] = None
    ) -> Mapping:
        request = {'model': self.to_dict()}
        if validation is not None:
            request['validation'] = validation.to_dict()
        return client.model.create(self.name, request, store)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            'name': self.name,
            'feature_set': self.feature_set.to_dict(),
            'model': {
                'type': self.type,
                'definition': self.definition,
            }
        }


# Utilities related to uploading models


def minimize_features(
    features: Sequence[StoredFeature],
    selected: Sequence[str]
) -> Sequence[StoredFeature]:
    """Reduce features to the minimum necessary

    The plugin will execute all features associated with a model, with no
    concern for if the model uses it or not. Minimize a list of stored features
    to only those required by the list of selected feature names.

    The plugin will execute features in the order given. Care is taken in the result
    to provide dependencies prior to their dependants. This order must be maintained
    when sending to the plugin.

    Parameters
    ----------
    features :
        List of stored features to select from
    selected:
        List of selected features that must be in the result

    Returns
    -------
        Minimal set of required features and their dependencies.
    """
    feat_map = {f.name: f for f in features}
    # Names are ordered from longest to shortest otherwise overlapping
    # names (eg: title and title_prefix) would always match the shorter
    # if it came first.
    names = sorted(feat_map.keys(), key=len, reverse=True)
    names_re = '|'.join(re.escape(name) for name in names)
    all_names_re = r'(?<!\w)({})(?!\w)'.format(names_re)
    matcher = re.compile(all_names_re)

    def deps(feature_name: str) -> Iterator[str]:
        feature = feat_map[feature_name]
        # TODO: Scripts can also have feature dependencies, but we don't (yet) support them.
        if feature.template_language == 'derived_expression':
            # Help mypy, we cant fit this in the type system without
            # subclassing per-template language
            assert isinstance(feature.template, str)
            for dep_name in matcher.findall(feature.template):
                yield from deps(dep_name)
        # This must come after the dependencies to ensure they are available.
        yield feature_name

    # Use only the keys OrderedDict() to effectively get an OrderedSet
    # implementation. This will return features in the order added, which
    # ensures all dependencies are satisfied.
    selected_features = cast(Dict[str, bool], OrderedDict())
    for feature_name in selected:
        for dep in deps(feature_name):
            selected_features[dep] = True

    return [feat_map[feature_name] for feature_name in selected_features.keys()]


class LtrModelUploader:
    # TODO: This constructor is massive, and this class is basically two functions and
    # some configuration. This should be representable in a better way.
    def __init__(
        self,
        elastic: Elasticsearch,
        model_name: str,
        model_type: str,
        model: Mapping,
        feature_source_definition: str,
        features: Sequence[str],
        validation: Optional[ValidationRequest]
    ) -> None:
        self.elastic = elastic
        self.ltr = LtrClient(elastic)
        self.model_name = model_name
        self.model_type = 'model/xgboost+json'  # todo: stop hardcoding
        self.model = model
        self.feature_source_definition = feature_source_definition
        # Break down def into it's pieces
        self.feature_def_type, self.feature_set_name, self.feature_store_name = \
            explode_ltr_model_definition(self.feature_source_definition)
        if self.feature_def_type != 'featureset':
            # This is actually a limitation of the forced featureset
            # minimization, although things could be abstracted to support
            # multiple paths.
            raise NotImplementedError('Can only derive featuresets from other featuresets currently')
        self.features = features
        self.validation = validation

    def _select_feature_set(self) -> StoredFeatureSet:
        """Determine the feature set to use for uploading.

        Returns the smallest possible feature set that can be used to
        provide the required features.
        """
        feature_set = StoredFeatureSet.get(
            self.ltr, self.feature_set_name, self.feature_store_name)
        missing = set(self.features).difference(feature_set.feature_names)
        if missing:
            raise Exception('Missing features in feature set: %s' % (', '.join(missing)))

        final_features = minimize_features(feature_set.features, self.features)

        # This feature set is never stored directly as a feature set, only as a
        # property of the model, so the name doesn't really matter.
        name = '{}-minimized'.format(feature_set.name)
        return StoredFeatureSet(name, final_features)

    def upload(self) -> StoredModel:
        if not self.ltr.store.exists(self.feature_store_name):
            raise Exception('Missing feature store [{}] on cluster [{}]'.format(self.feature_store_name, self.elastic))

        if self.ltr.model.exists(self.model_name, self.feature_store_name):
            raise Exception('A model named [{}] already exists on cluster [{}]'.format(self.model_name, self.elastic))

        feature_set = self._select_feature_set()
        model = StoredModel(self.model_name, feature_set, self.model_type, self.model)
        response = model.create(self.ltr, self.validation)
        # TODO: Not sure how we could get 2xx without created, but lets check anyways
        if response.get('result') != 'created':
            raise Exception(response)
        return model
