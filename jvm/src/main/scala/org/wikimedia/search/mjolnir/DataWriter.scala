package org.wikimedia.search.mjolnir

import java.io.{IOException, ObjectInputStream, ObjectOutputStream}
import java.nio.charset.StandardCharsets

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path => HDFSPath}
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row}

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.util.control.NonFatal


/**
  * Makes hadoop configuration serializable as a broadcast variable
  */
private class SerializableConfiguration(@transient var value: Configuration) extends Serializable {
  private def writeObject(out: ObjectOutputStream): Unit = tryOrIOException {
    out.defaultWriteObject()
    value.write(out)
  }

  private def readObject(in: ObjectInputStream): Unit = tryOrIOException {
    value = new Configuration(false)
    value.readFields(in)
  }

  private def tryOrIOException[T](block: => T): T = {
    try {
      block
    } catch {
      case e: IOException => throw e
      case NonFatal(e) => throw new IOException(e)
    }
  }
}

/**
 * counts the number of sequentially unique values it has seen.
 */
private class UniqueValueCounter[T]() {
  private var prev: T = _
  private var seen = 0

  def apply(value: T): Int = {
    if (value != prev) {
      prev = value
      seen += 1
    }
    seen
  }
}

/**
  * Write out a mjolnir dataframe from data_pipeline to hdfs as
  * txt files readable directly by xgboost and other libsvm based
  * parsers. Emitted files include the qid: extension used by
  * xgboost for ranking datasets. Emitted features are 1 indexed,
  * as some (?) parsers consider the label to be at 0.
  *
  * @param broadcastConfiguration Broadcasted hadoop configuration to access HDFS from executors.
  * @param sparse When true features with a value of zero are not emitted
  */
class DataWriter(
    broadcastConfiguration: Broadcast[SerializableConfiguration],
    sparse: Boolean = true
) extends Serializable {

  // Accepting JavaSparkContext for py4j compatability
  def this(sc: JavaSparkContext, sparse: Boolean) = this(sc.broadcast(new SerializableConfiguration(sc.hadoopConfiguration)), sparse)

  private def asHDFSPath(path: String): HDFSPath =
    if (path.charAt(0) == '/') {
      new HDFSPath(s"file://$path")
    } else {
      new HDFSPath(path)
    }

  // rdd contents for passing between formatPartition and writeOneFold
  private type OutputRow = (Int, Array[Byte])

  @transient lazy private val maybeSparsify =
    if (sparse) {
      features: Array[(Double, Int)] => features.filter(_._1 != 0D)
    } else {
      features: Array[(Double, Int)] => features
    }

  private def makeLine(label: Int, qid: Int, features: Vector): String = {
    val prefix = s"$label qid:$qid "
    // Some (?) parsers consider the label to be index 0, so + 1 the feature index
    // to make them 1-indexed.
    val stringifiedFeatures = maybeSparsify(features.toArray.zipWithIndex).map {
      case (feat, index) => s"${index + 1}:$feat"
    }
    stringifiedFeatures.mkString(prefix, " ", "\n")
  }

  @transient lazy private val utf8 = StandardCharsets.UTF_8

  private def foldFromRow(index: Int)(row: Row): Int = row.getLong(index).toInt

  private def formatPartition(schema: StructType, foldCol: Option[String])(rows: Iterator[Row]): Iterator[OutputRow] = {
    val chooseFold = foldCol
      .map(schema.fieldIndex)
      .map(foldFromRow)
      .getOrElse({ row: Row => 0 })

    val labelIndex = schema.fieldIndex("label")
    val featuresIndex = schema.fieldIndex("features")
    val queryIndex = schema.fieldIndex("query")
    val qid = new UniqueValueCounter[String]()

    for (row <- rows) yield {
      val fold = chooseFold(row)
      val line = makeLine(
        row.getInt(labelIndex),
        qid(row.getString(queryIndex)),
        row.getAs[Vector](featuresIndex))
      (fold, line.getBytes(utf8))
    }
  }

  // take nullable string and output java map for py4j compatability
  def write(df: DataFrame, pathFormat: String, foldCol: String, numFolds: Int): Array[java.util.Map[String, String]] = {
    //import collection.JavaConverters._
    import collection.JavaConverters.mapAsJavaMapConverter
    write(df, pathFormat, Option(foldCol), numFolds).map(_.asJava)
  }

  /**
    * @param df         Output from data_pipeline. Must have be repartitioned on query and sorted by query
    *                   within partitions. This should have a large number of partitions or later coalesce
    *                   will be imbalanced.
    * @param pathFormat Format for hdfs paths. Params are %s: name, %s: fold, %d: partition
    * @param foldCol    Long column to source which fold a row belongs to
    * @return List of folds each represented by a list of partitions each containing a map
    *         from split name to hdfs path for that partition.
    */
  def write(
    df: DataFrame,
    pathFormat: String,
    foldCol: Option[String],
    numFolds: Int
  ): Array[Map[String, String]] = {
    // Our qid:n numbers must be monotonic, essentially requiring them to
    // be computed on a single instance.
    val rdd = df.rdd.repartition(1).mapPartitions(formatPartition(df.schema, foldCol))

    def pathFormatter(name: String, foldId: String): HDFSPath = {
      asHDFSPath(pathFormat.format(name, foldId))
    }

    // Future.sequence requires an execution context
    import scala.concurrent.ExecutionContext.Implicits.global

    // Each fold task will write one copy of the complete dataset. Take
    // head of each fold since we know each fold has a single partition
    val folds = (0 until numFolds).map(writeOneFold(
      rdd, foldCol, _, numFolds, pathFormatter).map(_.head))

    Await.result(Future.sequence(folds), Duration.Inf).toArray
  }

  private def writeOneFold(
    rdd: RDD[OutputRow],
    foldCol: Option[String],
    fold: Int,
    numFolds: Int,
    pathFormatter: (String, String) => HDFSPath
  ): Future[Seq[Map[String, String]]] = {
    // Array mapping from zero indexed fold id to the named split to write that fold to
    val foldIdToNamedSplit = foldCol.map { name =>
      // If fold column provided assign that to test, everything else as train
      (0 until numFolds).map { x =>
        if (x == fold) "test" else "train"
      }.toArray
      // Otherwise everything in an "all" bucket. formatPartition ensures
      // that all rows report fold of 0L when foldCol is not provided.
    }.getOrElse(Array("all"))

    // Accepts a name and partition id, returns path to write out to
    def foldPathFormatter(name: String): HDFSPath = foldCol
      .map(_ => pathFormatter(name, fold.toString))
      .getOrElse(pathFormatter(name, "x"))

    // Per write method this has a single partition containing all data
    rdd.mapPartitions(writeOnePartition(
          foldPathFormatter, foldIdToNamedSplit))
      .collectAsync()
      .asInstanceOf[Future[Seq[Map[String, String]]]]
  }

  // Writes out a single partition of training data. One partition
  // may contain gigabytes of data so this should do as little work
  // as possible per-row.
  private def writeOnePartition(
    pathFormatter: String => HDFSPath,
    foldIdToNamedSplit: Array[String]
  )(
    rows: Iterator[OutputRow]
  ): Iterator[Map[String, String]] = {
    val (paths, namedWriters) = {
      val unique = foldIdToNamedSplit.toSet.toVector
      val paths = unique.map(pathFormatter)
      val files = paths.map(path => {
        val fs = path.getFileSystem(broadcastConfiguration.value.value)
        fs.create(path)
      })
      (unique.zip(paths.map(_.toString)).toMap, unique.zip(files).toMap)
    }

    val writers = foldIdToNamedSplit.map(namedWriters.apply)

    try {
      for ((fold, line) <- rows) {
        writers(fold).write(line)
      }
    } finally {
      namedWriters.values.foreach(_.close)
    }

    Iterator(paths)
  }

}
