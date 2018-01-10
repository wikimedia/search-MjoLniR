package org.wikimedia.search.mjolnir

import java.nio.file.{Files, Paths}

import org.apache.spark.api.java.JavaSparkContext

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

  test("basic training") {
    val base_path = { x: String => getClass.getResource(s"fixtures/datasets/$x") }
    val params = Map(
      "objective" -> "rank:ndcg",
      "eval_metric" -> "ndcg@5"
    )
    val fold = Seq("train", "test").map { name =>
      // We need to copy the resource files to the filesystem where xgboost can read them
      val remote_path = getClass.getResource(s"/fixtures/datasets/$name.txt")
      val remote_query_path = getClass.getResource(s"/fixtures/datasets/$name.txt.query")

      // Using createTempFile to get a base dir + random name
      val local_path = Files.createTempFile("mjolnir-test", "")
      val local_query_path = Paths.get(local_path + ".query")

      Files.delete(local_path)
      Files.copy(remote_path.openStream(), local_path)
      Files.copy(remote_query_path.openStream(), local_query_path)

      name -> ("file:" + local_path.toString)
    }.toMap

    try {
      val model = MlrXGBoost.trainWithFiles(
        spark.sparkContext, Array(fold), "train",
        params, numRounds = 5, earlyStoppingRound = 0)
      assert(model.summary.trainObjectiveHistory.length == 5)
      assert(model.summary.testObjectiveHistory.nonEmpty)
      assert(model.summary.testObjectiveHistory.get.length == 5)
      assert(model.booster.getModelDump().length == 5)
    } finally {
      fold.values.map { local_path =>
        Files.deleteIfExists(Paths.get(local_path))
        Files.deleteIfExists(Paths.get(local_path + ".query"))
      }
    }
  }
}
