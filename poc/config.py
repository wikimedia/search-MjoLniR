import os

from utils import os_utils

# Config for sourcing clicks
WIKI_PROJECT = 'en.wikipedia'
MIN_NUM_SEARCHES = 10
MAX_QUERIES = 150000

# Config for training DBN
DBN_CONFIG = {
    'MAX_ITERATIONS': 40,
    'DEBUG': False,
    'PRETTY_LOG': True,
    'MIN_DOCS_PER_QUERY': 10,
    'MAX_DOCS_PER_QUERY': 20,
    'SERP_SIZE': 20,
    'QUERY_INDEPENDENT_PAGER': False,
    'DEFAULT_REL': 0.5,
}

# Hadoop Directories
HADOOP_PREFIX = 'hdfs://analytics-hadoop/user/ebernhardson/ltr/%s_%dS_%dQ' % (
    WIKI_PROJECT, MIN_NUM_SEARCHES, MAX_QUERIES)
HADOOP_PREFIX_LOCAL = '/mnt/hdfs/user/ebernhardson/ltr/%s_%dS_%dQ' % (
    WIKI_PROJECT, MIN_NUM_SEARCHES, MAX_QUERIES)

# Initial result from sql queries sourcing click data
CLICK_DATA = '%s/click_data' % (HADOOP_PREFIX)

# feature logs as dataframe
FEATURE_LOGS = '%s/feature_logs' % (HADOOP_PREFIX)

# Data to feed into DBN
DBN_INPUT = '%s/dbn_input' % (HADOOP_PREFIX)
DBN_INPUT_LOCAL = '%s/dbn_input' % (HADOOP_PREFIX_LOCAL)
# Result data from DBN
DBN_RELEVANCE = '%s/dbn_output' % (HADOOP_PREFIX)

# Feature vectors sourced from elasticsearch ltr plugin about
# all the (query, hit_page_id) pairs in CLICK_DATA, merged
# with relevance scores from RELEVANCE_DATA
VECTOR_DATA = "%s/vector_data" % (HADOOP_PREFIX)

# Local Directories
ROOT_DIR = "../data/%s_%dS_%dQ" % (WIKI_PROJECT, MIN_NUM_SEARCHES, MAX_QUERIES)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TMP_DIR = os.path.join(ROOT_DIR, 'tmp')

# Data files

# Train/test/validation splits in svmrank format
SVMRANK_TRAIN_DATA = os.path.join(DATA_DIR, 'svmrank_train.txt')
SVMRANK_TEST_DATA = os.path.join(DATA_DIR, 'svmrank_test.txt')
SVMRANK_VALI_DATA = os.path.join(DATA_DIR, 'svmrank_vali.txt')

# Train/test/validation splits in lightgbm format
LIGHTGBM_TRAIN_DATA = os.path.join(HADOOP_PREFIX, 'lightgbm/train')
LIGHTGBM_TEST_DATA = os.path.join(HADOOP_PREFIX, 'lightgbm/test')
LIGHTGBM_VALI_DATA = os.path.join(HADOOP_PREFIX, 'lightgbm/vali')

# Train/test/validation splits for xgboost on hdfs
XGBOOST_TRAIN_DATA = os.path.join(HADOOP_PREFIX, 'xgboost/train')
XGBOOST_TEST_DATA = os.path.join(HADOOP_PREFIX, 'xgboost/test')
XGBOOST_VALI_DATA = os.path.join(HADOOP_PREFIX, 'xgboost/vali')

# Train/test/validations splits in a pickled dicct
VECTOR_DATA_SPLIT = os.path.join(DATA_DIR, 'vector_splits.pkl')

# Map from feature id to feature name
SVMRANK_LABELS = os.path.join(DATA_DIR, 'svmrank_labels.txt')

os_utils._create_dirs([
    DATA_DIR,
    TMP_DIR
])
