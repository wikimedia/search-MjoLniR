from argparse import ArgumentParser
from typing import Callable, Optional, List

from mjolnir.cli.helpers import Cli, HivePartition, feature_vector_stats
from mjolnir.transformations import feature_selection

from pyspark.sql import DataFrame


def transform(
    feature_vectors: HivePartition,
    labeled_query_page: HivePartition,
    num_features: int,
    wikis: Optional[List[str]],
    temp_dir: str,
    **kwargs
) -> DataFrame:
    if wikis is None:
        wikis = [r.wikiid for r in feature_vectors.df.select('wikiid').drop_duplicates().collect()]

    transformer = feature_selection.transformer(
        labeled_query_page.df, temp_dir, wikis, num_features)
    return transformer(feature_vectors.df)


def configure(parser: ArgumentParser) -> Callable:
    main = Cli('feature_selection', transform, parser)
    # I/O
    main.require_labeled_query_page_partition()
    main.require_feature_vectors_partition()
    main.require_output_table(
        [('feature_set', '{output_feature_set}')],
        metadata_fn=feature_vector_stats)
    main.require_temp_dir()

    # script parameterization
    main.add_argument('--output-feature-set', required=True)
    main.add_argument('--num-features', type=int, required=True)
    main.add_argument('--wikis', nargs='+', default=None)
    return main
