from __future__ import absolute_import
import argparse
from collections import defaultdict, OrderedDict
import glob
import hashlib
import json
import logging
import mjolnir.cirrus
import mjolnir.esltr
import mjolnir.featuresets
import mjolnir.utils
import os
import pickle
import pprint
import random
import re
import requests


MATCH_FEATURE_NAMES = re.compile(r'(?<!\w)(\w+)(?!\w)')


def feature_dependencies(feature_definitions, feature_name):
    deps = []
    feature = feature_definitions[feature_name]
    if feature['template_language'] == 'derived_expression':
        # Do a bad job at extracting names of features referenced by derived features
        for maybe_feature_name in MATCH_FEATURE_NAMES.findall(feature['template']):
            if maybe_feature_name in feature_definitions:
                deps += feature_dependencies(feature_definitions, maybe_feature_name)
    # This must come after the dependencies to ensure they are available.
    deps.append(feature_name)
    return deps


def make_minimal_feature_set(feature_definitions, features):
    """Reduce feature_definitions to the minimum necessary features

    Parameters
    ----------
    feature_definitions : list of dict
        Feature definitions from the elasticsearch LTR plugin
    features : list of str
        List of feature names required

    Returns
    -------
    list of dict
        feature_definitions filtered to the minimal necessary
    """
    feature_definitions = {f['name']: f for f in feature_definitions}
    # Using a dict to prevent duplicates. Inserting multiple times
    # does not change order.
    selected_features = OrderedDict()
    for feature in features:
        for dep in feature_dependencies(feature_definitions, feature):
            selected_features[dep] = feature_definitions[dep]
    return selected_features.values()


def minimal_feature_set_action(cluster, features, def_name, store_name, minimal_feature_set_name):
    cluster_base_path = random.choice(mjolnir.cirrus.SEARCH_CLUSTERS[cluster])
    ltr = mjolnir.esltr.Ltr(base_path=cluster_base_path)
    feature_store = ltr.get_feature_store(store_name)
    if not feature_store.exists():
        raise Exception('Missing feature store')
    feature_set = feature_store.get_feature_set(def_name)
    if not feature_set.exists():
        raise Exception('Missing feature set')

    feature_definitions = feature_set.list_features()['_source']['featureset']['features']
    feature_def_feature_names = set(f['name'] for f in feature_definitions)
    missing = features.difference(feature_def_feature_names)
    if missing:
        raise Exception('Missing features in feature set: %s' % (', '.join(missing)))

    if len(feature_definitions) == len(features):
        return 'existing', def_name
    else:
        # TODO: This doesn't apply the validation stage of feature creation
        minimal_feature_definitions = make_minimal_feature_set(feature_definitions, features)
        minimal_feature_set = feature_store.get_feature_set(minimal_feature_set_name)
        if not minimal_feature_set.exists():
            def create():
                print('Creating feature set %s on cluster %s' % (minimal_feature_set_name, cluster))
                minimal_feature_set.add_features(minimal_feature_definitions)

            return 'create', create
        else:
            existing = set(f['name'] for f in minimal_feature_set.list_features()['_source']['featureset']['features'])
            required = set(f['name'] for f in minimal_feature_definitions)
            if len(existing.union(required)) != len(required):
                raise Exception('Minimal feature set already exists but has wrong features')
            return 'existing', minimal_feature_set_name


def choose_feature_set(search_clusters, ltr_feature_definition, features):
    """Get or create a feature set for uploading models

    Parameters
    ----------
    ltr_feature_definition : str
        (featureset|model):name[@storename]
    features : iterable of str
        List of features required by the model. If this is a subset
        of those specified by ltr_feature_definition a new feature set
        will be created with only those features.
        This is necessary until the LTR plugin supports only evaluating
        the set of used features.

    Returns
    -------
    str
        Featureset name
    """
    def_type, def_name, store_name = mjolnir.utils.explode_ltr_model_definition(
        ltr_feature_definition)
    if def_type != 'featureset':
        raise Exception('TODO: Not Implemented')
    features = set(features)

    actions = defaultdict(list)
    features_hash = hashlib.md5('|'.join(features)).hexdigest()[:8]
    minimal_feature_set_name = '%s-minimal-%s' % (def_name, features_hash)
    # TODO Why not do this per cluster instead of all at once. Does it matter?
    for cluster in search_clusters:
        action, params = minimal_feature_set_action(cluster, features, def_name, store_name, minimal_feature_set_name)
        actions[action].append(params)

    if len(actions) != 1:
        raise Exception('TODO: Not Implemented')

    action = actions.keys()[0]
    if action == 'create':
        for fn in actions['create']:
            fn()
        feature_set_name = minimal_feature_set_name
    elif action == 'existing':
        names = set(a[0] for a in actions['existing'])
        if len(names) != 1:
            raise Exception('feature set name differs between clusters')
        feature_set_name = actions['existing'][0]
    else:
        raise Exception('Unreached')

    return store_name, feature_set_name


def upload_models(input_dir, search_clusters, wikis, yes):
    fnames = glob.glob('%s/tune_*.pickle' % (input_dir))
    wikis = set(wikis)
    for fname in fnames:
        print(fname)
        with open(fname, 'rb') as f:
            tune = pickle.load(f)
        wiki = tune['metadata']['wiki']
        if wikis and wiki not in wikis:
            continue

        model_fname = os.path.join(input_dir, 'model_%s.json' % (wiki))
        with open(model_fname, 'rb') as f:
            model = json.load(f)

        # Take the portion of the directory name before the first _ as portion of model name
        name = os.path.basename(os.path.dirname(fname)).split('_')[0]
        # TODO: v1?
        model_name = '%s_%s_v1' % (name, wiki)

        answer = 'y' if yes else input('\tUpload model %s? ' % (model_name)).lower()
        if answer != 'y':
            print('\tSkipping.')
            print('')
            continue

        req = {
            "model": {
                "name": model_name,
                "model": {
                    "type": "model/xgboost+json",
                    "definition": model
                },
            },
            "validation": {
                "index": wiki,
                # TODO: hard coded params..
                "params": {
                    "query_string": "example query string"
                }
            }
        }

        # Handle per-wiki feature variation due to feature selection
        if 'wiki_features' in tune['metadata']['dataset']:
            wiki_features = tune['metadata']['dataset']['wiki_features'][wiki]
        else:
            wiki_features = tune['metadata']['dataset']['features']

        feature_store_name, feature_set_name = choose_feature_set(
            search_clusters,
            tune['metadata']['dataset']['feature_definitions'],
            wiki_features)
        url_pattern = '%%s/_ltr/%s_featureset/%s/_createmodel' % (
            '' if feature_store_name is None else feature_store_name + '/',
            feature_set_name)

        for cluster in search_clusters:
            base_url = random.choice(mjolnir.cirrus.SEARCH_CLUSTERS[cluster])
            print('\tUploading to %s' % (base_url))
            res = requests.post(url_pattern % (base_url,), data=json.dumps(req),
                                headers={'Content-Type': 'application/json'})

            try:
                parsed = res.json()
                if 'result' in parsed and parsed['result'] == 'created':
                    print('Created on %s!' % (cluster))
                else:
                    pprint.pprint(parsed)
            except ValueError:
                print(res.text)


def arg_parser():
    parser = argparse.ArgumentParser(description='Upload models to elasticsearch')
    parser.add_argument(
        '-i', '--input', dest='input_dir', required=True,
        help='Input directory generated by training_pipeline to upload models from')
    parser.add_argument(
        '-c', '--search-cluster', metavar='cluster', dest='search_clusters', type=str,
        action='append', choices=mjolnir.cirrus.SEARCH_CLUSTERS.keys(),
        help='List of clusters to upload models to')
    # TODO: This doesn't really work with our config based spark runner which expects all
    # parameters to be a k/v pair.
    parser.add_argument(
        '-y', '--yes', dest='yes', action='store_true', default=False,
        help='Don\'t prompt on uploads, assume yes to all models.')
    parser.add_argument(
        'wikis', metavar='wikis', type=str, nargs='*',
        help='Wikis to upload models for. Empty for all models in input directory')
    return parser


def main(**kwargs):
    upload_models(**kwargs)


if __name__ == "__main__":
    logging.basicConfig()
    kwargs = dict(vars(arg_parser().parse_args()))
    main(**kwargs)
