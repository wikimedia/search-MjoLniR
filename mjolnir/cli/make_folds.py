from argparse import ArgumentParser
from typing import Callable, List, Mapping

from mjolnir.cli.helpers import Cli, HivePartition
import mjolnir.transform as mt
from mjolnir.transformations import make_folds


def transform(
    date: str,
    output_path: str,
    feature_vectors: HivePartition,
    labeled_query_page: HivePartition,
    wiki: List[str],
    num_folds: int,
    **kwargs
) -> Mapping:
    transformer = make_folds.transformer(
        labeled_query_page.df, wiki, output_path, num_folds)
    df_out = transformer(feature_vectors.df)

    # At this point we have a quite small dataframe containing paths
    # to the much larger datasets. We could stuff these into a hive table,
    # but any training will need to collect this all to the driver anyways.
    # Putting it in a hive table forces training to initialize spark before
    # looking at stats about the dataset, making it harder to size the executors.
    # Instead we are simply going to write .json files into hdfs and call it a day.

    # TODO: This will simply fail if the output file already exists. Any re-run
    # of this will need to clear out this file along with any related datasets
    # that it includes paths to.
    mt.check_schema(df_out, mt.TrainingFiles)
    return {
        'partition_spec': {
            'date': date,
            'labeling_algorithm': labeled_query_page.partition_value('algorithm'),
            'feature_set': feature_vectors.partition_value('feature_set'),
            'wikiid': wiki,
        },
        'metadata': {
            'wikiid': wiki,
            'num_obs': feature_vectors.metadata['num_obs'][wiki],
            'features': feature_vectors.metadata['wiki_features'][wiki],
        },
        'rows': [row.asDict() for row in df_out.collect()],
    }


def configure(parser: ArgumentParser) -> Callable:
    main = Cli('make_folds', transform, parser)
    # I/O
    main.require_feature_vectors_partition()
    main.require_labeled_query_page_partition()
    main.require_output_metadata()
    # Script parameterization
    main.add_argument('--num-folds', type=int, default=5)
    main.add_argument('--wiki', required=True)
    return main
