"""
Example script demonstrating the data collection portion of the MLR pipeline.
This is mostly to demonstrate how everything ties together

To run:
    PYSPARK_PYTHON=venv/bin/python spark-submit \
        --jars /path/to/mjolnir-with-dependencies.jar
        --artifacts 'mjolnir_venv.zip#venv' \
        --files /usr/lib/libhdfs.so.0.0.0
        mjolnir/cli/data_pipeline.py
"""

import mjolnir.dbn
import mjolnir.metrics
import mjolnir.features
import mjolnir.sampling
from pyspark import SparkContext
from pyspark.sql import HiveContext
from pyspark.sql import functions as F


def main(sc, sqlContext):
    sqlContext.sql("ADD JAR /mnt/hdfs/wmf/refinery/current/artifacts/refinery-hive.jar")
    sqlContext.sql("CREATE TEMPORARY FUNCTION stemmer AS 'org.wikimedia.analytics.refinery.hive.StemmerUDF'")

    # TODO: Should be CLI option
    wikis = ['enwiki', 'dewiki', 'ruwiki', 'frwiki']

    # Load click data from HDFS
    df_clicks = (
        sqlContext.read.parquet(
            'hdfs://analytics-hadoop/wmf/data/discovery/query_clicks/daily/year=*/month=*/day=*')
        # Limit to the wikis we are working against
        .where(mjolnir.sampling._array_contains(F.array(map(F.lit, wikis)), F.col('wikiid')))
        # Clicks and hits contains a bunch of useful debugging data, but we don't
        # need any of that here. Save a bunch of memory by only working with
        # lists of page ids
        .withColumn('hit_page_ids', F.col('hits.pageid'))
        .drop('hits')
        .withColumn('click_page_ids', F.col('clicks.pageid'))
        .drop('clicks')
        # Normalize queries using the lucene stemmer
        .withColumn('norm_query', F.expr('stemmer(query, substring(wikiid, 1, 2))')))

    # Sample to some subset of queries per wiki
    df_sampled = (
        mjolnir.sampling.sample(
            df_clicks,
            wikis=wikis,
            seed=54321,
            queries_per_wiki=20000,
            min_sessions_per_query=10)
        # Explode source into a row per displayed hit
        .select('*', F.expr("posexplode(hit_page_ids)").alias('hit_position', 'hit_page_id'))
        .drop('hit_page_ids')
        # Mark all hits that were clicked by a user
        .withColumn('clicked', F.expr('array_contains(click_page_ids, hit_page_id)'))
        .drop('click_page_ids')
        .cache())

    # Learn relevances
    df_rel = (
        mjolnir.dbn.train(df_sampled, {
            'MAX_ITERATIONS': 40,
            'DEBUG': False,
            'PRETTY_LOG': True,
            'MIN_DOCS_PER_QUERY': 10,
            'MAX_DOCS_PER_QUERY': 20,
            'SERP_SIZE': 20,
            'QUERY_INDEPENDENT_PAGER': False,
            'DEFAULT_REL': 0.5})
        # naive conversion of relevance % into a label
        .withColumn('label', (F.col('relevance') * 10).cast('int')))

    df_all_hits = (
        df_sampled
        .select('wikiid', 'query', 'norm_query', 'hit_page_id', 'session_id', 'hit_position')
        .join(df_rel, how='inner', on=['wikiid', 'norm_query', 'hit_page_id'])
        .cache())

    # materialize df_all_hits and drop df_sampled
    df_all_hits.count()
    df_sampled.unpersist()

    # TODO: Training is per-wiki, should this be as well?
    weightedNdcgAt10 = mjolnir.metrics.ndcg(df_all_hits, 10, query_cols=['wikiid', 'query', 'session_id'])
    print 'weighted ndcg@10: %.4f' % (weightedNdcgAt10)

    df_hits = (
        df_all_hits
        .groupBy('wikiid', 'query', 'norm_query', 'hit_page_id')
        # weight is now the number of times a hit was displayed to a user
        .agg(F.count(F.lit(1)).alias('weight'),
             F.mean('hit_position').alias('hit_position'),
             # These should be the same per group, but to keep things easy
             # take first rather than grouping
             F.first('label').alias('label'),
             F.first('relevance').alias('relevance'))
        .cache())

    # materialize df_hits and drop df_all_hits
    df_hits.count()
    df_all_hits.unpersist()

    # TODO: Training is per-wiki, should this be as well?
    ndcgAt10 = mjolnir.metrics.ndcg(df_hits, 10, query_cols=['wikiid', 'query'])
    print 'unweighted ndcg@10: %.4f' % (ndcgAt10)

    # Collect features for all known queries. Note that this intentionally
    # uses query and NOT norm_query. Merge those back into the source hits.
    df_features = mjolnir.features.collect(
        df_hits,
        url_list=['http://elastic%d.codfw.wmnet:9200/_msearch' % (i) for i in range(2001, 2035)],
        indices={wiki: '%s_content' % (wiki) for wiki in wikis},
        feature_definitions=mjolnir.features.enwiki_features())
    df_hits_with_features = (
        df_hits
        .join(df_features, how='inner', on=['wikiid', 'query', 'hit_page_id'])
        .withColumn('labels', mjolnir.spark.add_meta(sc, F.col('labels'), {
            'weightedNdcgAt10': weightedNdcgAt10,
            'ndcgAt10': ndcgAt10,
        })))

    df_hits_with_features.write.parquet('hdfs://analytics-hadoop/user/ebernhardson/mjolnir/hits_with_features')


if __name__ == "__main__":
    sc = SparkContext(appName="MLR: data collection pipeline")
    # spark info logging is incredibly spammy. Use warn to have some hope of
    # human decipherable output
    sc.setLogLevel('WARN')
    sqlContext = HiveContext(sc)
    main(sc, sqlContext)
