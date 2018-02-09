"""Interface for elasticesarch ltr plugin management api

Doesn't try too hard, simply wraps some url/method handling so we dont have to
duplicate it elsewhere. All responses are return as-is from the management API.
Responses from elasticsearch containing an 'error' key are converted to
exceptions.

FIXME: template parameters are awkward
"""

from __future__ import absolute_import
import os
import requests


class Ltr(object):
    def __init__(self, host='localhost', port=9200, session_factory=None):
        self.host = host
        self.port = port
        self.session = session_factory() if session_factory else requests.Session()
        self.base_path = 'http://%s:%d/_ltr/' % (host, port)

    def list_feature_stores(self):
        return self.request('GET')

    def get_feature_store(self, name=None):
        return LtrFeatureStore(self, name)

    def request(self, method, path='', **kwargs):
        url = os.path.join(self.base_path, path)
        res = self.session.request(method, url, **kwargs).json()
        if 'error' in res:
            e = Exception("LTR failure: " + str(res['error']))
            e.res = res
            raise e
        return res


class LtrFeatureStore(object):
    def __init__(self, ltr, name=None):
        self.ltr = ltr
        self.name = name
        if name:
            if name[0] == '_':
                raise ValueError('Invalid name [%s]: must not start with _' % (name))
            if '/' in name:
                raise ValueError('Invalid name [%s]: cannot contain /' % (name))
            self.base_path = name
        else:
            self.base_path = ''
            self.name = '_default_'

    def fullname(self):
        return self.name

    def create(self):
        return self.request('PUT')

    def delete(self):
        return self.request('DELETE')

    def exists(self):
        res = self.request('GET')
        if self.name == '_default_' and 'stores' in res and self.name in res['stores']:
            return True
        elif 'exists' in res and res['exists']:
            return True
        else:
            return False

    def list_features(self, prefix=None):
        return self._list('_features/', prefix)

    def list_feature_sets(self, prefix=None):
        return self._list('_featureset/', prefix)

    def _list(self, path, prefix):
        params = {}
        if prefix:
            params['prefix'] = prefix
        return self.request('GET', path, params=params)

    def get_feature_set(self, name, validation=None):
        return LtrFeatureSet(self, name, validation)

    def request(self, method, path='', **kwargs):
        combined_path = os.path.join(self.base_path, path)
        return self.ltr.request(method, combined_path, **kwargs)


class LtrFeatureSet(object):
    def __init__(self, store, name, validation=None):
        self.store = store
        if name[0] == '_':
            raise ValueError('Invalid name [%s]: must not start with _' % (name))
        if '/' in name:
            raise ValueError('Invalid name [%s]: cannot contain /' % (name))
        self.name = name
        self.base_path = '_featureset/%s' % (name)
        self.validation = validation

    def fullname(self):
        return '%s.%s' % (self.store.fullname(), self.name)

    def create(self, features):
        req = {}
        if features:
            req['features'] = features
        if self.validation:
            req['validation'] = self.validation

        return self.request('PUT', json=req)

    def delete(self):
        return self.request('DELETE')

    def exists(self):
        res = self.request('GET')
        return 'found' in res and res['found']

    def list_features(self):
        return self.request('GET')

    def add_features(self, features):
        req = {'features': features}
        if self.validation:
            req['validation'] = self.validation
        return self.request('POST', '_addfeatures', json=req)

    def request(self, method, path='', **kwargs):
        combined_path = os.path.join(self.base_path, path)
        return self.store.request(method, combined_path, **kwargs)
