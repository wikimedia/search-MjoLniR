import logging
import os
import yaml

SEARCH_CONFIG_PATHS = ('./', '/etc/mjolnir')
log = logging.getLogger(__name__)


def decide_config_path(config_path, default_filename):
    if config_path is not None:
        return config_path
    for path in (os.path.join(d, default_filename) for d in SEARCH_CONFIG_PATHS):
        if os.path.exists(path):
            return path
    return None


def load_config(config_path, default_filename):
    config_path = decide_config_path(config_path, default_filename)
    if config_path is None:
        return None
    log.info('Loading configuration from %s', config_path)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
