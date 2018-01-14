from __future__ import absolute_import
import os
import mjolnir.dbn


def test_dbn_train(hive_context, fixtures_dir):
    df = hive_context.read.json(os.path.join(fixtures_dir, "dbn_input.json"))
    labeled = mjolnir.dbn.train(df, {
        # Don't use this config for prod, it's specifically for small testing
        'MIN_DOCS_PER_QUERY': 1,
        'MAX_DOCS_PER_QUERY': 4,
        'DEFAULT_REL': 0.5,
        'MAX_ITERATIONS': 1,
        'GAMMA': 0.9,
    })
    assert len(labeled.columns) == 4
    assert 'wikiid' in labeled.columns
    assert 'norm_query_id' in labeled.columns
    assert 'hit_page_id' in labeled.columns
    assert 'relevance' in labeled.columns

    # Make sure we didn't add/drop data somewhere
    data = labeled.collect()
    assert len(data) == 8, "Expecting 4 relevance labels * 2 queries in fixtures"

    # Make sure wikiid is kept through the process
    wikiids = set([row.wikiid for row in data])
    assert len(wikiids) == 1
    assert u'foowiki' in wikiids

    # Make sure the set of unique queries is kept
    queries = set([row.norm_query_id for row in data])
    assert len(queries) == 2
    assert 12345 in queries
    assert 23456 in queries

    # Make sure the dbn is provided data in the right order, by looking at what comes out
    # at the top and bottom of each query. This should also detect if something went wrong
    # with partitioning, causing parts of a query to train in separate DBN's
    test = sorted([row for row in data if row.norm_query_id == 12345], key=lambda row: row.relevance, reverse=True)
    assert test[0].hit_page_id == 1111
    assert test[3].hit_page_id == 3333

    zomg = sorted([row for row in data if row.norm_query_id == 23456], key=lambda row: row.relevance, reverse=True)
    assert zomg[0].hit_page_id == 4444
    assert zomg[3].hit_page_id == 1111
    # page 1111 should have been skipped every time, resulting in a very low score
    assert zomg[3].relevance == 0.1
