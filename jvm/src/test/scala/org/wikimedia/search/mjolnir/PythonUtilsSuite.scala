package org.wikimedia.search.mjolnir

class PythonUtilsSuite extends SharedSparkContext {
  test("query groups should maintain input order") {
    val sqlContext = spark.sqlContext
    import sqlContext.implicits._
    val df = spark.sparkContext.parallelize(List(
      "foo", "foo", "bar", "baz", "baz", "baz"
    )).repartition(1).toDF("queryId")

    val res = PythonUtils.calcQueryGroups(df, "queryId")
    assert(res == Seq(Seq(2, 1, 3)))
  }

  test("query groups should match partition count") {
    val sqlContext = spark.sqlContext
    import sqlContext.implicits._
    val rdd1 = spark.sparkContext.parallelize(List(
      "foo", "bar", "bar"
    )).repartition(1)
    val rdd2 = spark.sparkContext.parallelize(List(
      "baz", "baz", "bang", "hi", "hi", "hi"
    )).repartition(1)

    val df = spark.sparkContext.union(rdd1, rdd2).toDF("queryId")
    val res = PythonUtils.calcQueryGroups(df, "queryId")
    assert(res == Seq(Seq(1, 2), Seq(2, 1, 3)))
  }
}
