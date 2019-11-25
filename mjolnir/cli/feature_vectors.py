from argparse import ArgumentParser
from typing import Callable, List, Optional

from mjolnir.cli.helpers import Cli, HivePartition, feature_vector_stats
import mjolnir.sampling
import mjolnir.transform as mt
from mjolnir.transformations import feature_vectors

from pyspark.sql import DataFrame, functions as F


@mt.typed_transformer(mt.QueryClicks, mt.QueryPage)
def resample_clicks_to_query_page(
    df_cluster: DataFrame,
    random_seed: Optional[int],
    samples_per_wiki: int
) -> mt.Transformer:
    # Resamples the click log by proxy of resampling clusters, such
    # that a complete cluster is either included or excluded from the
    # resulting dataset.
    # TODO: Evaluate alternative resampling, such as perhaps only dropping from
    # clusters where all clicks were to the top result (implying an "easy" search).

    mt.check_schema(df_cluster, mt.QueryClustering)
    return mt.seq_transform([
        # Grab only the parts of the query log we need to make the resulting sampled QueryPage
        lambda df: df.select('query', 'wikiid', 'session_id', 'hit_page_ids'),
        mt.join_cluster_by_query(df_cluster),
        # [1] is because sample returns a tuple of (page_counts, df)
        mt.temp_rename_col('cluster_id', 'norm_query_id', lambda df: mjolnir.sampling.sample(
            df, random_seed, samples_per_wiki)[1]),
        lambda df: df.withColumn(
            'page_id', F.explode('hit_page_ids')).drop('hit_page_ids')
    ])


def transform(
    query_clicks: HivePartition,
    query_clustering: HivePartition,
    samples_per_wiki: int,
    random_seed: Optional[int],
    wikis: List[str],
    brokers: str,
    topic_request: str,
    topic_response: str,
    feature_set: str,
    **kwargs
) -> DataFrame:
    transformer = mt.seq_transform([
        mt.restrict_wikis(wikis),
        resample_clicks_to_query_page(
            query_clustering.df, random_seed, samples_per_wiki),
        feature_vectors.transformer(
            brokers, topic_request, topic_response, feature_set)
    ])
    return transformer(query_clicks.df)


def configure(parser: ArgumentParser) -> Callable:
    main = Cli('feature_vectors', transform, parser)
    # I/O
    main.require_kafka_arguments()
    main.require_query_clicks_partition()
    main.require_query_clustering_partition()
    main.require_output_table(
        [('feature_set', '{feature_set}')],
        metadata_fn=feature_vector_stats)

    # script parameterization
    main.add_argument('--feature-set', required=True)
    main.add_argument('--samples-per-wiki', type=int, default=10000000)
    main.add_argument('--random-seed', type=int, default=None)
    main.add_argument('--wikis', nargs='+', default=None)

    return main
