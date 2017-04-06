import config
from utils import spark_utils

def prepare_data(hive):
    hive.sql("CREATE TEMPORARY FUNCTION stemmer AS 'org.wikimedia.analytics.refinery.hive.StemmerUDF'")

    # Choose a random selection of queries that have more than N results
    rand_selection = hive.sql("""
       SELECT
           x.project, x.norm_query
       FROM (
           SELECT
               project,
               STEMMER(query, SUBSTR(wikiid, 1, 2)) as norm_query,
               count(distinct year, month, day, session_id) as num_searchs
           FROM
               discovery.query_clicks_daily
           WHERE
               year >= 2016
               AND project = '%s'
           GROUP BY
               project,
               STEMMER(query, SUBSTR(wikiid, 1, 2))
           ) x
       WHERE
           x.num_searchs >= %d
       DISTRIBUTE BY
           rand()
       SORT BY
           rand()
       LIMIT
           %d
    """ % (config.WIKI_PROJECT, config.MIN_NUM_SEARCHES, config.MAX_QUERIES))

    hive.registerDataFrameAsTable(rand_selection, "rand_selection")

    # Find all the data for our random set of queries
    rand_set = hive.sql("""
        SELECT
            TRIM(query_clicks_daily.query) as query,
            STEMMER(query_clicks_daily.query, SUBSTR(query_clicks_daily.wikiid, 1, 2)) AS norm_query,
            query_clicks_daily.timestamp as search_timestamp,
            query_clicks_daily.wikiid,
            query_clicks_daily.hits,
            query_clicks_daily.clicks,
            CONCAT_WS('_', query_clicks_daily.session_id, CAST(year AS string), CAST(month AS string), CAST(day AS string)) AS session_id
        FROM
            discovery.query_clicks_daily
        JOIN
            rand_selection
        ON
            rand_selection.norm_query = STEMMER(query_clicks_daily.query, SUBSTR(query_clicks_daily.wikiid, 1, 2))
            AND rand_selection.project = query_clicks_daily.project
        WHERE
            year >= 2016
    """)
    hive.registerDataFrameAsTable(rand_set, "rand_set")

    # Break down the data into per-hit data. This is needed because a single session
    # can see the same hit for a query multiple times, as we re-run the search when a
    # user does: click -> read, unsatisfied -> go back -> click different result
    per_hit = hive.sql("""
        SELECT
            rand_set.query,
            rand_set.norm_query,
            rand_set.search_timestamp,
            rand_set.wikiid,
            rand_set.session_id,
            click.pageid as click_page_id,
            click.timestamp as click_timestamp,
            hit.title as hit_title,
            hit.pageid as hit_page_id,
            hit.score as hit_score,
            hit_position
        FROM
            rand_set
        LATERAL VIEW
            EXPLODE(rand_set.clicks) c as click
        LATERAL VIEW
            POSEXPLODE(rand_set.hits) h as hit_position, hit
    """)
    hive.registerDataFrameAsTable(per_hit, "per_hit")

    # re-group the per-hit data into sessions, this time with a single data point per-hit
    return hive.sql("""
        SELECT
            query,
            norm_query,
            session_id,
            hit_page_id,
            hit_title,
            AVG(hit_score) AS hit_score,
            AVG(hit_position) AS hit_position,
            ARRAY_CONTAINS(COLLECT_LIST(click_page_id), hit_page_id) AS clicked
        FROM
            per_hit
        GROUP BY
            query,
            norm_query,
            session_id,
            hit_page_id,
            hit_title
    """)

if __name__ == "__main__":
    sc, hive = spark_utils._init("LTR: prepare")

    # Write out a raw copy of data we will work with
    prepare_data(hive).write.parquet(config.CLICK_DATA)
