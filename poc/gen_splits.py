from pyspark.sql.types import IntegerType
from pyspark.sql.functions import UserDefinedFunction

import config
from utils import spark_utils

MAX_EXECUTORS = 400
NUM_PARTITIONS = 800

def rel_to_label(relevance):
    if relevance > 0.9:
        return 4
    elif relevance > 0.75:
        return 3
    elif relevance > 0.5:
        return 2
    elif relevance > 0.3:
        return 1
    else:
        return 0

def split_partition(iterator):
    train_count = 0
    test_count = 0
    vali_count = 0
    splits = []
    # starting at 1 prevents div by zero
    processed = 1
    for row in iterator:
        # count is an unusable name directly because there is also a count
        # function on tuple, which Row extends.
        count = row.asDict()['count']
        if float(train_count) / processed < .6:
            splits.append((row.norm_query, 0))
            train_count += count
        elif float(test_count) / processed < .2:
            splits.append((row.norm_query, 1))
            test_count += count
        elif float(vali_count) / processed < .2:
            splits.append((row.norm_query, 2))
            vali_count += count
        else:
            splits.append((row.norm_query, 0))
            train_count += count
        processed += count
    return splits

def write_split(split, fname):
    codec = 'org.apache.hadoop.io.compress.GzipCodec'
    def write_one(col):
        (split
            .map(lambda row: row.__getattr__(col))
            .saveAsTextFile("%s/%s" % (fname, col), codec)
        )

    write_one('data')
    write_one('weight')
    write_one('qid')

def main():
    sc, hive = spark_utils._init("LTR: gen splits", {
        "spark.sql.shuffle.partitions": NUM_PARTITIONS,
        "spark.dynamicAllocation.maxExecutors": MAX_EXECUTORS,
    })

    vector_data = (hive.read.parquet(config.VECTOR_DATA)
        .repartition(NUM_PARTITIONS)
        # This persist saves a couple shuffles and disk reads.
        # minor win but seems worthwhile
        .persist()
    )

    # We don't want randomized splits, we want to make sure that each
    # norm_query is only within a single split, which makes this a bit
    # annoying. Additionally we would like our 60/20/20 split to have an
    # equal number of rows, rather than an equal number of norm_query
    # values.
    splits = (vector_data
        .groupBy('norm_query')
        .count()
        # Sorting ensures consistent results for future runs
        .sort('norm_query')
        # partitions need to be a minimum size for splits to be balanced
        .coalesce(100)
        .mapPartitions(split_partition)
        .toDF(['norm_query', 'split'])
    )

    query_ids = (vector_data
        .select('query')
        .drop_duplicates()
        .rdd.map(lambda r: r.query)
        .zipWithUniqueId()
        .toDF(['query', 'query_id'])
    )

    # sort feature cols so we have a simple way to figure out the
    # feature name -> feature id map later
    feature_cols = sorted(vector_data.columns)
    feature_cols.remove('query')
    feature_cols.remove('norm_query')
    feature_cols.remove('hit_page_id')
    feature_cols.remove('relevance')
    feature_cols.remove('weight')

    pre_split = (vector_data
        .join(query_ids, 'query', how='inner')
        .join(splits, 'norm_query', how='inner')
        .map(lambda row: ((
            "%d %s" % (
                rel_to_label(row.relevance),
                " ".join(['%d:%f' % (1+i, row.__getattr__(feat))
                          for i, feat in enumerate(feature_cols)])),
            row.weight,
            row.query_id,
            row.split
        )))
        .toDF(['data', 'weight', 'qid', 'split'])
        # rows with same qid need to be output together, sorting seems
        # a reasonable way to make that happen.
        .sort('qid')
        # The data size isn't actually that big by this point, cut down to
        # 20 partitions to write out to disk.
        # @TODO All packages we feed this into only support a single input
        # file, perhaps instead we should use .toLocalIterator() and write
        # them all out locally?
        .coalesce(20)
        # we need to write out 9 separate things from this one result, persist
        # so we don't re-do the above work which is pretty expensive
        .persist()
    )

    train = pre_split.filter(pre_split.split == 0)
    test = pre_split.filter(pre_split.split == 1)
    vali = pre_split.filter(pre_split.split == 2)

    write_split(train, config.LIGHTGBM_TRAIN_DATA)
    write_split(test, config.LIGHTGBM_TEST_DATA)
    write_split(vali, config.LIGHTGBM_VALI_DATA)


if __name__ == "__main__":
    main()

