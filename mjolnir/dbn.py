"""
Implements training a Dynamic Bayesian Network, using the clickmodels library,
within spark
"""

from __future__ import absolute_import
from clickmodels.inference import DbnModel
from clickmodels.input_reader import InputReader
import json
import pyspark.sql
from pyspark.sql import functions as F
from pyspark.sql import types as T
import mjolnir.spark


def _deduplicate_hits(session_hits):
    """Deduplicate multiple views of a hit by a single session.

    A single session may have seen the same result list multiple times, for
    example by clicking a link, then clicking back and clicking a second link.
    Normalize that data together into a single record per hit_page_id even if
    it was displayed to a session multiple times.

    Parameters
    ----------
    session_hits : list
        A list of hits seen by a single session.

    Returns
    -------
    list
        List of hits shown to a session de-duplicated to contain only one entry
        per hit_page_id.
    """
    by_hit_page_id = {}
    for hit in session_hits:
        if hit.hit_page_id in by_hit_page_id:
            by_hit_page_id[hit.hit_page_id].append(hit)
        else:
            by_hit_page_id[hit.hit_page_id] = [hit]

    deduped = []
    for hit_page_id, hits in by_hit_page_id.iteritems():
        hit_positions = []
        clicked = False
        for hit in hits:
            hit_positions.append(hit.hit_position)
            clicked |= bool(hit.clicked)
        deduped.append(pyspark.sql.Row(
            hit_page_id=hit_page_id,
            hit_position=sum(hit_positions) / float(len(hit_positions)),
            clicked=clicked))
    return deduped


def _gen_dbn_input(iterator):
    """Converts an iterator over spark rows into the DBN input format.

    It is perhaps undesirable that we serialize into a string with json so
    InputReader can deserialize, but it is not generic enough to avoid this
    step.

    Parameters
    ----------
    iterator : ???
        iterator over pyspark.sql.Row. Each row must have a wikiid,
        norm_query_id, and list of hits each containing hit_position,
        hit_page_id and clicked.

    Yields
    -------
    string
        Line for a single item of the input iterator formatted for use
        by clickmodels InputReader.
    """
    for row in iterator:
        results = []
        clicks = []
        deduplicated = _deduplicate_hits(row.hits)
        deduplicated.sort(key=lambda hit: hit.hit_position)
        for hit in deduplicated:
            results.append(str(hit.hit_page_id))
            clicks.append(hit.clicked)
        yield '\t'.join([
            '0',  # unused identifier
            str(row.norm_query_id),  # group the session belongs to
            row.wikiid,  # region
            '0',  # intent weight
            json.dumps(results),  # hits displayed in session
            json.dumps([False] * len(results)),  # layout (unused)
            json.dumps(clicks)  # Was result clicked
        ])


def _extract_labels_from_dbn(model, reader):
    """Extracts all learned labels from the model.

    Paramseters
    -----------
    model : clickmodels.inference.DbnModel
        A trained DBN model
    reader : clickmodels.input_reader.InputReader
        Reader that was used to build the list of SessionItem's model was
        trained with.

    Returns
    -------
    list of tuples
        List of four value tuples each containing wikiid, norm_query_id,
        hit_page_id and relevance.
    """
    # reader converted all the page ids into an internal id, flip the map so we
    # can change them back. Not the most memory efficient, but it will do.
    uid_to_url = {uid: url for url, uid in reader.url_to_id.iteritems()}
    rows = []
    for (norm_query_id, wikiid), qid in reader.query_to_id.iteritems():
        # clickmodels required the group key to be a string, convert back
        # to an int to match input data
        norm_query_id = int(norm_query_id)
        for uid, data in model.urlRelevances[False][qid].iteritems():
            relevance = data['a'] * data['s']
            hit_page_id = int(uid_to_url[uid])
            rows.append((wikiid, norm_query_id, hit_page_id, relevance))
    return rows


def train(df, dbn_config, num_partitions=200):
    """Generate relevance labels for the provided dataframe.

    Process the provided data frame to generate relevance scores for
    all provided pairs of (wikiid, norm_query_id, hit_page_id). The input
    DataFrame must have a row per hit_page_id that was seen by a session.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        User click logs with columns wikiid, norm_query_id, session_id,
        hit_page_id, hit_position, clicked.
    dbn_config : dict
        Configuration needed by the DBN. See clickmodels documentation for more
        information.
    num_partitions : int
        The number of partitions to split input data into for training.
        Training will load the entire partition into python to feed into the
        DBN, so a large enough number of partitions need to be used that we
        don't blow out executor memory.

    Returns
    -------
    spark.sql.DataFrame
        DataFrame with columns wikiid, norm_query_id, hit_page_id, relevance.
    """
    mjolnir.spark.assert_columns(df, ['wikiid', 'norm_query_id', 'session_id',
                                      'hit_page_id', 'hit_position', 'clicked'])

    def train_partition(iterator):
        """Learn the relevance labels for a single DataFrame partition.

        Before applying to a partition ensure that sessions for queries are not
        split between multiple partitions.

        Parameters
        ----------
        iterator : iterator over pyspark.sql.Row's.

        Returns
        -------
        list of tuples
            List of (wikiid, norm_query_id, hit_page_id, relevance) tuples.
        """
        reader = InputReader(dbn_config['MIN_DOCS_PER_QUERY'],
                             dbn_config['MAX_DOCS_PER_QUERY'],
                             False,
                             dbn_config['SERP_SIZE'],
                             False,
                             discard_no_clicks=True)
        sessions = reader(_gen_dbn_input(iterator))
        dbn_config['MAX_QUERY_ID'] = reader.current_query_id + 1
        model = DbnModel((0.9, 0.9, 0.9, 0.9), config=dbn_config)
        model.train(sessions)
        return _extract_labels_from_dbn(model, reader)

    rdd_rel = (
        df
        # group and collect up the hits for individual (wikiid, norm_query_id,
        # session_id) tuples to match how the dbn expects to receive data.
        .groupby('wikiid', 'norm_query_id', 'session_id')
        .agg(F.collect_list(F.struct('hit_position', 'hit_page_id', 'clicked')).alias('hits'))
        # Partition into small batches ensuring that all matching (wikiid,
        # norm_query_id) rows end up on the same partition.
        # TODO: The above groupby and this repartition both cause a shuffle, is
        # it possible to make that a single shuffle? Could push the final level
        # of grouping into python, but that could just as well end up worse?
        .repartition(num_partitions, 'wikiid', 'norm_query_id')
        # Run each partition through the DBN to generate relevance scores.
        .rdd.mapPartitions(train_partition))

    # Using toDF() is very slow as it has to run some of the partitions to check their
    # types, and then run all the partitions later to get the actual data. To prevent
    # running twice specify the schema we expect.
    return df.sql_ctx.createDataFrame(rdd_rel, T.StructType([
        T.StructField('wikiid', T.StringType(), False),
        T.StructField('norm_query_id', T.LongType(), False),
        T.StructField('hit_page_id', T.LongType(), False),
        T.StructField('relevance', T.DoubleType(), False)
    ]))
