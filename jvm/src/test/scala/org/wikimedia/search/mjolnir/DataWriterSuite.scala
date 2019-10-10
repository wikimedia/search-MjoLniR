package org.wikimedia.search.mjolnir

import java.nio.file.{Files, Path, Paths}

import org.apache.commons.io.FileUtils
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.types.MetadataBuilder

import scala.util.Random
import org.apache.spark.sql.{functions => F}
import org.scalatest.prop.TableDrivenPropertyChecks._

class DataWriterSuite extends SharedSparkContext {
  private def randInt(max: Int) = F.udf(() => Random.nextInt(max))
  // TODO: Why are these longs?
  private def randLong(max: Int) = F.udf(() => Random.nextInt(max).toLong)
  private def randVec(n: Int) = F.udf(() => Vectors.dense(Array.fill(n)(Random.nextDouble)))
  private def randString(items: Seq[String]) = F.udf(() => items(Random.nextInt(items.length)))

  private val numFolds = 3
  private def makeData() = {
    val queries = Array.fill(30)(Random.nextString(10))
    val meta = new MetadataBuilder().putLong("num_folds", numFolds).build()
    spark.sqlContext.range(1000)
      .withColumn("label", randInt(4)())
      .withColumn("features", randVec(5)())
      .withColumn("fold", randLong(numFolds)().as("fold", meta))
      .withColumn("query", randString(queries)())
      // Must have more partitions than numFolds above
      // or coalesce won't do anything.
      .repartition(numFolds * 2, F.col("query"))
      .sortWithinPartitions("query")
  }

  test("Write out various standard fold configs as text files") {
    forAll(Table(
      ("fold_col", "expectedFolds", "expectedSplits"),
      (None, 1, Set("all")),
      (Some("fold"), numFolds, Set("train", "test"))
    )) { (foldCol, numFolds, expectedSplits) =>
      val testDir = Files.createTempDirectory("mjolnir-test")
      try {
        val df = makeData()
        val pattern = s"$testDir/%s-fold-%s"
        val writer = new DataWriter(spark.sparkContext, sparse = false)
        val folds = writer.write(df, pattern, foldCol, numFolds)

        assert(folds.length == numFolds)
        folds.foreach { fold =>
          // Items in partition but not expected
          assert(fold.keySet.diff(expectedSplits).isEmpty)
          // Items expected but not in partition
          assert(expectedSplits.diff(fold.keySet).isEmpty)
          fold.values.foreach { path =>
            // Paths should actually exist
            assert(Files.exists(Paths.get(path.substring("file:".length))))
          }
        }
      } finally {
        FileUtils.deleteDirectory(testDir.toFile)
      }
    }
  }
}
