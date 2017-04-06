import json

import config
from utils import spark_utils


PREFIX_LEN = len("[2017-01-11 19:15:15,020][INFO ][org.wikimedia.search.ltr.FeatureLogger] ")

def main():
    sc, hive = spark_utils._init("LTR: Parse Logs")

    rawFeatureLogs = (sc.textFile(config.RAW_FEATURE_LOGS)
        # Strip the prefix and parse the json
        .map(lambda line: json.loads(line[PREFIX_LEN:]))
    )

    featureList = rawFeatureLogs.take(1)[0]['vec'].keys()

    (rawFeatureLogs.map(lambda log: [
            int(log['_id']),
            log['_marker'],
        ] + [log['vec'][featName] for featName in featureList])
        .toDF(['hit_page_id', 'query'] + featureList)
        .write.parquet(config.FEATURE_LOGS)
    )


if __name__ == "__main__":
    main()
