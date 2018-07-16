"""
Collect hit page ids for queries from elasticsearch
"""

from __future__ import absolute_import
import json
import mjolnir.cirrus
import mjolnir.spark
import random
import requests


def _make_es_query(row, top_n):
    return {
        "_source": False,
        "stats": [
            "mjolnir",
        ],
        "size": top_n,
        "query": mjolnir.cirrus.full_text_query(row.query),
        "rescore": [mjolnir.cirrus.rescore()],
    }


def _create_bulk_query(rows, indices, top_n):
    bulk_query = []
    for row in rows:
        if row.wikiid in indices:
            index = indices[row.wikiid]
        else:
            # Takes advantage of aliases for the wikiid typically used by
            # CirrusSearch
            index = row.wikiid
        bulk_query.append('{"index": "%s"}' % (index))
        bulk_query.append(json.dumps(_make_es_query(row, top_n)))
    return "%s\n" % ('\n'.join(bulk_query))


def _handle_response(response):
    assert response.status_code == 200
    parsed = response.json()
    assert 'responses' in parsed, response.text
    for one_response in parsed['responses']:
        yield [int(hit['_id']) for hit in one_response['hits']['hits']]


def _batch(iterable, n):
    cur_batch = []
    for x in iterable:
        cur_batch.append(x)
        if len(cur_batch) >= n:
            yield cur_batch
            cur_batch = []
    if len(cur_batch) > 0:
        yield cur_batch


def transform(df, url_list, indices=None, batch_size=15, top_n=5, session_factory=requests.Session):
    """Collect hit page ids for queries from elasticsearch

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    url_list : list of str
        List of urls for elasticsearch servers
    indices : dict, optional
        Map from wikiid to the elasticsearch index to query. If not provided the wikiid
        will be used as the index name.
    batch_size : int
        Number of queries to issue in a single multi-search
    top_n : int
        Number of hits to collect per query
    session_factory : object

    Returns
    -------
    pyspark.sql.DataFrame
    """
    mjolnir.spark.assert_columns(df, ['wikiid', 'query', 'norm_query'])
    if indices is None:
        indices = {}

    def collect_partition_hit_page_ids(rows):
        # mjolnir.cirrus.make_request will modify the passed url list as hosts are rejected.
        # Make a copy so those changes don't escape.
        partition_url_list = list(url_list)
        random.shuffle(partition_url_list)
        with session_factory() as session:
            for batch_rows in _batch(rows, batch_size):
                bulk_query = _create_bulk_query(batch_rows, indices, top_n)
                response = mjolnir.cirrus.make_request('msearch', session, partition_url_list, bulk_query)
                for row, hit_page_ids in zip(batch_rows, _handle_response(response)):
                    # Extend the provided row with an extra field. Ideally we would
                    # instead use a UDF, but that makes re-using a requests session
                    # difficult. Explicit, rather than row + (hit_page_ids,) to ensure
                    # ordering matches toDF([...]) call that names them
                    yield (row.wikiid, row.query, row.norm_query, hit_page_ids)

    # To protect the cluster from overload limit the # of partitions.
    # elasticsearch issues bulk queries in parallel so we are running, at most,
    # batch_size * num_executors queries at a time within the cluster. The
    # value of 1500 here has shown to be reasonable to keep the cluster busy
    # but out of thread pool rejection.
    mjolnir.cirrus.check_idle(url_list, session_factory)
    max_executors = 1500 / batch_size
    if df.rdd.getNumPartitions() > max_executors:
        df = df.coalesce(max_executors)

    return (
        df
        .rdd.mapPartitions(collect_partition_hit_page_ids)
        .toDF(['wikiid', 'query', 'norm_query', 'hit_page_ids']))
