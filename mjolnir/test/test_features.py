import mjolnir.features
import pyspark.sql


def test_collect_es(spark_context, hive_context, make_requests_session):
    def session_factory():
        return make_requests_session('requests/test_features.sqlite3')

    r = pyspark.sql.Row('wikiid', 'query', 'hit_page_id')
    source_data = {
        'apple': [18978754, 36071326, 856],
        'foo': [11178, 1140775, 844613]
    }
    rows = [r('enwiki', query, page_id) for query, ids in source_data.items() for page_id in ids]
    df = spark_context.parallelize(rows).toDF()

    df_result = mjolnir.features.collect_es(df, ['http://localhost:9200/_msearch'],
                                            mjolnir.features.enwiki_features(),
                                            {'enwiki': 'enwiki_content'},
                                            session_factory=session_factory)
    result = df_result.collect()
    feature_names = df_result.schema['features'].metadata['features']
    expected_page_ids = set([row.hit_page_id for row in rows])
    result_page_ids = set([row.hit_page_id for row in result])
    assert expected_page_ids == result_page_ids

    apple_856_features = [row.features for row in result if row.query == 'apple' and row.hit_page_id == 856][0]
    apple_856 = dict(zip(feature_names, apple_856_features.toArray()))
    assert apple_856['title'] == 36.30035
    assert apple_856['file_text'] == 0.0
