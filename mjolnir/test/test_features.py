from __future__ import absolute_import
from collections import OrderedDict
import mjolnir.features
import numpy as np
import pyspark.sql


def test_collect_ltr_plugin(spark_context, make_requests_session):
    def session_factory():
        return make_requests_session('requests/test_features.sqlite3')

    r = pyspark.sql.Row('wikiid', 'query', 'hit_page_id')
    source_data = {
        'apple': [18978754, 36071326, 856],
        'foo': [11178, 1140775, 844613]
    }
    rows = [r('enwiki', query, page_id) for query, ids in source_data.items() for page_id in ids]
    df = spark_context.parallelize(rows).toDF()

    accu = df._sc.accumulator(OrderedDict(), mjolnir.features.FeatureNamesAccumulator())
    df_result = mjolnir.features.collect_from_ltr_plugin(
        df, ['http://localhost:9200'],
        "featureset:enwiki_v1",
        accu,
        {'enwiki': 'enwiki_content'},
        session_factory=session_factory)

    result = df_result.collect()

    # all features must have been logged
    assert len(set(accu.value.values())) == 1
    feature_names = accu.value.keys()

    expected_page_ids = set([row.hit_page_id for row in rows])
    result_page_ids = set([row.hit_page_id for row in result])
    assert expected_page_ids == result_page_ids

    apple_856_features = [row.features for row in result if row.query == 'apple' and row.hit_page_id == 856][0]
    apple_856 = dict(zip(feature_names, apple_856_features.toArray()))
    for field in ('title', 'auxiliary_text'):
        assert type(apple_856[field]) == np.float64
        assert apple_856[field] > 0
    assert apple_856['file_text'] == 0.0
