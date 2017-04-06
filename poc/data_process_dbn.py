from pyspark import SparkContext
from pyspark.sql import HiveContext

import tempfile
import json
import codecs

import clickmodels
from clickmodels.inference import DbnModel
from clickmodels.input_reader import InputReader, SessionItem

import config
from utils import spark_utils

def train_dbn_partition():
    # Extra wrapper is necessary to ensure we don't try and
    # import config in the worker node
    DBN_CONFIG = config.DBN_CONFIG
    def work(iterator):
        # Dump iterator into a temp file
        reader = InputReader(DBN_CONFIG['MIN_DOCS_PER_QUERY'],
                             DBN_CONFIG['MAX_DOCS_PER_QUERY'],
                             False,
                             DBN_CONFIG['SERP_SIZE'],
                             False,
                             discard_no_clicks=True)

        # Evil hax to make our temporary file read/write utf-8,
        # as the queries contain utf-8
        f = tempfile.TemporaryFile()
        info = codecs.lookup('utf-8')
        f = codecs.StreamReaderWriter(f, info.streamreader, info.streamwriter, 'struct')
        for row in iterator:
            results = []
            clicks = []
            for hit in sorted(row.hits, key=lambda hit: hit.position):
                results.append(str(hit.page_id))
                clicks.append(bool(hit.clicked))
            f.write('\t'.join([
                "0", # unused identifier
                row.norm_query,
                "0", # region
                "0", # intent weight
                json.dumps(results), # displayed hits
                json.dumps([False] * len(results)), # layout
                json.dumps(clicks) # clicks
            ]) + "\n")
        f.seek(0)
        sessions = reader(f)
        del f


        dbn_config = DBN_CONFIG.copy()
        dbn_config['MAX_QUERY_ID'] = reader.current_query_id + 1
        # Test with a single iteration 
        #dbn_config['MAX_ITERATIONS'] = 1
        model = DbnModel((0.9, 0.9, 0.9, 0.9), config=dbn_config)
        model.train(sessions)

        relevances = []
        uid_to_url = dict((uid, url) for url, uid in reader.url_to_id.iteritems())
        for (query, region), qid in reader.query_to_id.iteritems():
            for uid, data in model.urlRelevances[False][qid].iteritems():
                relevances.append([query, int(uid_to_url[uid]), data['a'] * data['s']])

        return relevances
    return work

def session_to_dbn(row):
    results = []
    clicks = []
    for hit in sorted(row.hits, key=lambda hit: hit.position):
        results.append(str(hit.page_id))
        clicks.append(bool(hit.clicked))

    return [row.norm_query, results, clicks]

def prep_dbn(hive):
    hive.read.parquet(config.CLICK_DATA).registerTempTable('click_data')

    hive.sql("""
        SELECT
            norm_query,
            session_id,
            hit_page_id,
            AVG(hit_position) AS hit_position,
            ARRAY_CONTAINS(COLLECT_LIST(clicked), true) as clicked
        FROM
            click_data
        GROUP BY
            norm_query,
            session_id,
            hit_page_id
    """).registerTempTable('click_data_by_session')

    return (hive.sql("""
            SELECT
                norm_query,
                COLLECT_LIST(NAMED_STRUCT(
                    'position', hit_position,
                    'page_id', hit_page_id,
                    'clicked', clicked
                )) AS hits
            FROM
                click_data_by_session
            GROUP BY
                norm_query,
                session_id
        """)
        # Sort guarantees all sessions for same query
        # are in same partition
        .sort('norm_query')
    )


def main():
    sc, hive = spark_utils._init("LTR: DBN")

    # Attach clickmodels .egg. Very bold assumption it's in an egg...
    #clickmodels_path = clickmodels.__file__
    #clickmodels_egg_path = clickmodels_path[:clickmodels_path.find('.egg')+4]
    #sc.addPyFile(clickmodels_egg_path)

    prep_dbn(hive) \
        .mapPartitions(train_dbn_partition()) \
        .toDF(['norm_query', 'hit_page_id', 'relevance']) \
        .write.parquet(config.DBN_RELEVANCE)

if __name__ == "__main__":
    main()
