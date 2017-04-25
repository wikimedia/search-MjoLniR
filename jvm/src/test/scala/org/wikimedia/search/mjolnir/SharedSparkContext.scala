package org.wikimedia.search.mjolnir

import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfter, BeforeAndAfterAll, FunSuite}

trait SharedSparkContext extends FunSuite with BeforeAndAfter with BeforeAndAfterAll
  with Serializable {

  @transient protected implicit var spark: SparkSession = null

  override def beforeAll() {
    spark = SparkSession.builder()
      .appName("MjoLniRSuite")
      .master("local[*]")
      .config("spark.testing.memory", 536870912)
      .config("spark.driver.memory", "512m")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
  }

  override def afterAll() {
    if (spark != null) {
      spark.close()
    }
  }
}
