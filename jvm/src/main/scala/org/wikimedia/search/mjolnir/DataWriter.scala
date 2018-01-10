package org.wikimedia.search.mjolnir

import java.nio.charset.StandardCharsets

import org.apache.hadoop.fs.{Path => HDFSPath}
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row}

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}

/**
  * Write out a mjolnir dataframe from data_pipeline to hdfs as
  * txt files readable directly by xgboost/lightgbm. While not
  * explicitly called out in the return values there is a matching
  * path + ".query" file for each output file containing sequential
  * query counts needed by xgboost and lightgbm.
  *
  * @param broadcastConfiguration Broadcasted hadoop configuration to access HDFS from executors.
  * @param sparse When true features with a value of zero are not emitted
  */
class DataWriter(
    broadcastConfiguration: Broadcast[SerializableConfiguration],
    sparse: Boolean = true
) extends Serializable {

  // Accepting JavaSparkContext for py4j compatability
  def this(sc: JavaSparkContext) = this(sc.broadcast(new SerializableConfiguration(sc.hadoopConfiguration)))

  private def asHDFSPath(path: String): HDFSPath = if (path.charAt(0) == '/') {
    new HDFSPath(s"file://$path")
  } else {
    new HDFSPath(path)
  }

  // rdd contents for passing between formatPartition and writeOneFold
  private type OutputRow = (Int, Array[Byte], Array[Byte])

  // Writes out a single partition of training data. One partition
  // may contain gigabytes of data so this should do as little work
  // as possible per-row.
  private def writeOneFold(
                            pathFormatter: (String, Int) => HDFSPath,
                            config: Array[String]
                          )(partitionId: Int, rows: Iterator[OutputRow]): Iterator[Map[String, String]] = {
    // .toSet.toVector gives us a unique list, but feels like hax
    val paths = config.toSet.toVector.map { name: String =>
      name -> pathFormatter(name, partitionId)
    }

    val writers = paths.map { case (name: String, path: HDFSPath) =>
      val fs = path.getFileSystem(broadcastConfiguration.value.value)
      val a = fs.create(path)
      val b = fs.create(new HDFSPath(path + ".query"))
      name -> (a, b)
    }.toMap

    try {
      for ((fold, line, queryLine) <- rows) {
        val out = writers(config(fold))
        out._1.write(line)
        out._2.write(queryLine)
      }
    } finally {
      writers.values.foreach({ out =>
        out._1.close()
        out._2.close()
      })
    }

    Iterator(paths.map { case (k, v) => k -> v.toString }.toMap)
  }

  @transient lazy private val maybeSparsify = if (sparse) {
    { features: Array[(Double, Int)] => features.filter(_._1 != 0D) }
  } else {
    { features: Array[(Double, Int)] => features }
  }

  private def makeLine(features: Vector, label: Int): String =
    maybeSparsify(features.toArray.zipWithIndex).map {
      case (feat, index) => s"${index + 1}:$feat";
    }.mkString(s"$label ", " ", "\n")

  // Counts sequential queries and emits boundaries
  private def queryCounter(): (String, Option[String]) => String = {
    var count = 0;
    { (query: String, nextQuery: Option[String]) =>
      count += 1
      if (nextQuery.exists(_.equals(query))) {
        // next matches, keep the count going
        ""
      } else {
        // next query is different. zero out
        val out = s"$count\n"
        count = 0
        out
      }
    }
  }

  @transient lazy private val utf8 = StandardCharsets.UTF_8

  private def formatPartition(schema: StructType, foldCol: Option[String])(rows: Iterator[Row]): Iterator[OutputRow] = {
    val chooseFold = foldCol.map { name =>
      val index = schema.fieldIndex(name);
    { row: Row => row.getLong(index).toInt }
    }.getOrElse({ row: Row => 0 })

    val labelIndex = schema.fieldIndex("label")
    val featuresIndex = schema.fieldIndex("features")
    val queryIndex = schema.fieldIndex("query")
    val makeQueryLine = queryCounter()
    val it = rows.buffered
    for (row <- it) yield {
      val fold = chooseFold(row)
      val nextQuery = if (it.hasNext) {
        Some(it.head.getString(queryIndex))
      } else {
        None
      }

      val line = makeLine(row.getAs[Vector](featuresIndex), row.getInt(labelIndex))
      val queryLine = makeQueryLine(row.getString(queryIndex), nextQuery)
      (fold, line.getBytes(utf8), queryLine.getBytes(utf8))
    }
  }

  // take nullable string and output java map for py4j compatability
  def write(df: DataFrame, numWorkers: Int, pathFormat: String, foldCol: String): Array[Array[java.util.Map[String, String]]] = {
    //import collection.JavaConverters._
    import collection.JavaConverters.mapAsJavaMapConverter
    write(df, numWorkers, pathFormat, Option(foldCol)).map(_.map(_.asJava))
  }

  /**
    * @param df         Output from data_pipeline. Must have be repartitioned on query and sorted by query
    *                   within partitions.
    * @param numWorkers The number of partitions each data file will be emitted as
    * @param pathFormat Format for hdfs paths. Params are %s: name, %s: fold, %d: partition
    * @param foldCol    Long column to source which fold a row belongs to
    * @return List of folds each represented by a list of partitions each containing a map
    *         from split name to hdfs path for that partition.
    */
  def write(
             df: DataFrame,
             numWorkers: Int,
             pathFormat: String,
             foldCol: Option[String]
           ): Array[Array[Map[String, String]]] = {
    val rdd = df.rdd.mapPartitions(formatPartition(df.schema, foldCol))

    val numFolds = foldCol.map { name => df.schema(name).metadata.getLong("num_folds").toInt }.getOrElse(1)

    try {
      // Materialize rdd so the parallel tasks coming up share the result.otherwise spark can just
      // coalesce all the work above into the minimal number of output workers and repeat it for
      // each partition it writes out.
      // TODO: depends on if we have enough nodes to cache the data. On a busy cluster maybe not...
      rdd.cache()
      rdd.count()

      val folds = (0 until numFolds).map { fold =>
        // TODO: It might be nice to just accept Seq[Map[Int, String]] that lists folds
        // each map is a config parameter. Pushes naming up the call chain.
        val config = foldCol.map { name =>
          // If fold column provided assign that to test, everything else as train
          (0 until numFolds).map { x =>
            if (x == fold) "test" else "train"
          }.toArray
          // Otherwise everything in an "all" bucket. formatPartition ensures
          // that all rows report fold of 0L when foldCol is not provided.
        }.getOrElse(Array("all"))

        // Accepts a name and partition id, returns path to write out to
        val pathFormatter: (String, Int) => HDFSPath = foldCol.map { _ =>
          val foldId = fold.toString;
        { (name: String, partition: Int) => asHDFSPath(pathFormat.format(name, foldId, partition)) }
          // If all in one bucket the indicate fold in path with 'x'
        }.getOrElse({ (name, partition) => asHDFSPath(pathFormat.format(name, "x", partition)) })

        rdd.coalesce(numWorkers)
          .mapPartitionsWithIndex(writeOneFold(pathFormatter, config))
          .collectAsync()
          .asInstanceOf[Future[Seq[Map[String, String]]]]
      }
      // Future.sequence requires an execution context
      import scala.concurrent.ExecutionContext.Implicits.global
      Await.result(Future.sequence(folds), Duration.Inf).toArray.map(_.toArray)
    } finally {
      rdd.unpersist()
    }
  }
}
