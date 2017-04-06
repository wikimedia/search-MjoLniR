from pyspark import SparkContext, SparkConf
from pyspark.sql import HiveContext

def _init(appName, conf=None):
    if type(conf) == dict:
        properties = conf
        conf = SparkConf()
        for k, v in properties.iteritems():
            conf.set(k, v)
    sc = SparkContext(appName=appName, conf=conf)
    sc.setLogLevel("WARN")
    sc.addFile('hdfs://analytics-hadoop/user/hive/hive-site.xml')
    hive = HiveContext(sc)

    return (sc, hive)
