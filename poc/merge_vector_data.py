import pyspark.sql.functions as F

import config
from utils import spark_utils

def main():
    sc, hive = spark_utils._init("LTR: Merge")

    dfSource = hive.read.parquet(config.CLICK_DATA)
    # Convert duplicates into weights
    dfWeight = (dfSource
        .groupBy('query', 'hit_page_id')
        .agg(
            dfSource.query, 
            dfSource.hit_page_id,
            F.count('query').alias('weight')
        )
    )

    dfFeature = hive.read.parquet(config.FEATURE_LOGS)
    dfRelevance = hive.read.parquet(config.DBN_RELEVANCE)

    (dfSource
        .select('query', 'norm_query', 'hit_page_id')
        .dropDuplicates()
        .join(dfRelevance, on=['norm_query', 'hit_page_id'], how='inner')
        .join(dfWeight, on=['query', 'hit_page_id'], how='inner')
        .join(dfFeature, on=['query', 'hit_page_id'], how='inner')
        .write.parquet(config.VECTOR_DATA)
    )




if __name__ == "__main__":
    main()
