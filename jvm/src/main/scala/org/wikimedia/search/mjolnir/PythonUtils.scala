package org.wikimedia.search.mjolnir

import org.apache.spark.ml.feature.{LabeledPoint => MLLabeledPoint}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vectors, Vector => MLVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.types.DoubleType
import scala.collection.mutable.ListBuffer

/**
  * Utilities to help bridge the gap between xgboost4j-scala and
  * pyspark.
  */
object PythonUtils {
  private def shiftVector(vec: MLVector): MLVector = vec match {
    case y: DenseVector => Vectors.dense(Array(0D) ++ y.toArray)
    case y: SparseVector => Vectors.sparse(y.size + 1, y.indices.map(_ + 1), y.values)
  }

  /**
   * There is no access to LabeledPoint from pyspark, but various methods such as
   * trainWithRDD and eval require an RDD[MLLabeledPoint]. This offers a bridge to
   * convert a Dataset into the required format.
   *
   * @param ds Input dataframe containing features and label
   * @param featureCol Name of the column containing feature vectors
   * @param labelCol Name of the column containing numeric labels
   * @param shiftRight Shift all features to index + 1. This is a disapointing hack,
   *                  but due to the way data files are created feature indices start
   *                  at 1 and the 0 feature is empty. This allows to shift to match
   *                  when evaluating a dataframe againts a model trained that way.
   */
  def toLabeledPoints(ds: Dataset[_], featureCol: String, labelCol: String, shiftRight: Boolean): RDD[MLLabeledPoint] = {
    ds.select(col(featureCol), col(labelCol).cast(DoubleType)).rdd.map {
      case Row(feature: MLVector, label: Double) =>
        val shiftedFeature = if (shiftRight) shiftVector(feature) else feature
        MLLabeledPoint(label, shiftedFeature)
    }
  }

  def toLabeledPoints(ds: Dataset[_], featureCol: String, labelCol: String): RDD[MLLabeledPoint] = {
    toLabeledPoints(ds, featureCol, labelCol, shiftRight = false)
  }


  /**
   * Training/evaluating a ranking model in XGBoost requires rows for the same
   * query to be provided sequentially, and it needs to know for each partition
   * what the sequential counts are. This proved to be quite expensive in python
   * with multiple double serialization steps, and is much more performant here.
   *
   * @param ds Dataset pre-partitioned and sorted by queryIdCol
   * @param queryIdCol Column to count sequential rows over
   */
  def calcQueryGroups(ds: Dataset[_], queryIdCol: String): Seq[Seq[Int]] = {
    val groupData = ds.select(col(queryIdCol)).rdd.mapPartitionsWithIndex {
      (partitionId, rows) =>
        rows.foldLeft(List[(String, Int)]()) {
          (acc, e) =>
            val queryId = e.getAs[String](queryIdCol)
            acc match {
              // If the head matches current queryId increment it
              case ((`queryId`, count) :: xs) => (queryId, count + 1) :: xs
              // otherwise add new item to head
              case _ => (queryId, 1) :: acc
            }
        // Replace query id with partition id
        }.map {
          case (_, count) => (partitionId, count)
        // reverse because the list was built backwards
        }.reverse.toIterator
    }.collect()

    val numPartitions = ds.rdd.getNumPartitions
    val groups = Array.fill(numPartitions)(ListBuffer[Int]())
    // Convert our list of (partitionid, count) pairs into the result
    // format. Spark guarantees to sort order has been maintained.
    for (e <- groupData) {
      groups(e._1) += e._2
    }
    groups
  }
}
