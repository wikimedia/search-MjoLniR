import hashlib
import mjolnir.features
import os
import pyspark.sql
import requests
import sqlite3


class MockSession(object):
    def __init__(self, fixture_file):
        self._session = None
        if fixture_file[0] != '/':
            dir_path = os.path.dirname(os.path.realpath(__file__))
            fixture_file = os.path.join(dir_path, fixture_file)
        # Use sqlite for storage so we don't have to figure out how
        # multiple pyspark executors write to the same file
        self.sqlite = sqlite3.connect(fixture_file)
        self.sqlite.execute(
            "CREATE TABLE IF NOT EXISTS requests " +
            "(digest text PRIMARY KEY, status_code int, content text)")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get(self, url, data=None):
        md5 = hashlib.md5()
        md5.update(url)
        md5.update(data)
        digest = md5.hexdigest()

        for row in self.sqlite.execute("SELECT status_code, content from requests WHERE digest=?", [digest]):
            return MockResponse(row[0], row[1])

        r = requests.get(url, data=data)

        try:
            self.sqlite.execute("INSERT INTO requests VALUES (?,?,?)", [digest, r.status_code, r.text])
            self.sqlite.commit()
        except sqlite3.IntegrityError:
            # inserted elsewhere? no big deal
            pass

        return MockResponse(r.status_code, r.text)


class MockResponse(object):
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


def session_factory():
    return MockSession('fixtures/requests/test_features.sqlite3')


def test_collect(spark_context, hive_context):
    r = pyspark.sql.Row('wikiid', 'query', 'hit_page_id')
    source_data = {
        'apple': [18978754, 36071326, 856],
        'foo': [11178, 1140775, 844613]
    }
    rows = [r('enwiki', query, page_id) for query, ids in source_data.items() for page_id in ids]
    df = spark_context.parallelize(rows).toDF()

    df_result = mjolnir.features.collect(df, ['http://localhost:9200/_msearch'],
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
