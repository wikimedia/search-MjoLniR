package org.wikimedia.search.mjolnir

import java.io.{ByteArrayInputStream, IOException}

import ml.dmlc.xgboost4j.java.{IRabitTracker, Rabit, RabitTracker => PyRabitTracker}
import ml.dmlc.xgboost4j.scala.spark.{MjolnirUtils, XGBoostModel}
import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost}
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.{FutureAction, Partitioner, SparkContext, TaskContext}
import org.apache.spark.rdd.RDD

/**
  * Rabit tracker configurations.
  *
  * @param workerConnectionTimeout The timeout for all workers to connect to the tracker.
  *                                Set timeout length to zero to disable timeout.
  *                                Use a finite, non-zero timeout value to prevent tracker from
  *                                hanging indefinitely (in milliseconds)
  *                                (supported by "scala" implementation only.)
  * @param trackerImpl Choice between "python" or "scala". The former utilizes the Java wrapper of
  *                    the Python Rabit tracker (in dmlc_core), whereas the latter is implemented
  *                    in Scala without Python components, and with full support of timeouts.
  *                    The Scala implementation is currently experimental, use at your own risk.
  */
case class TrackerConf(workerConnectionTimeout: Long, trackerImpl: String)

object TrackerConf {
  def apply(): TrackerConf = TrackerConf(0L, "python")
}

/**
  * Training helper for xgboost4j. Brings files from hdfs to the local executor
  * and runs training against them.
  *
  * @param asLocalFile Serializable helper class to copy files from hdfs to local
  */
private class MlrXGBoost(asLocalFile: AsLocalFile) extends Serializable {
  private def createDMatrices[A](
      inputs: Map[String, String]
  ): Map[String, DMatrix] = inputs.map { case (name, path) =>
    name -> asLocalFile(path) { localPath: String =>
      // DMatrix will read the file in on creation, and asLocalFile
      // will delete that local file. after creation of the dmatrix.
      new DMatrix(localPath)
    }
  }

  private[mjolnir] def buildDistributedBoosters(
      rdd: RDD[Map[String, String]],
      trainMatrix: String,
      params: Map[String, Any],
      rabitEnv: Option[java.util.Map[String, String]],
      numRounds: Int,
      earlyStoppingRound: Int = 0
  ): RDD[(Array[Byte], Map[String, Array[Float]])] =
    rdd.mapPartitions({ rows=>
      // XGBoost refuses to load our binary format if rabit has been
      // initialized, so we do it early. This make the odd situation
      // where we need to dispose of them before rabit is shutdown.
      rabitEnv.foreach { env =>
        env.put("DMLC_TASK_ID", TaskContext.getPartitionId().toString)
        Rabit.init(env)
      }
      try {
        // TODO: This doesn't guarantee delete when one dmatrix
        // fails init after another has been initialized.
        val watches = createDMatrices(rows.next())
        try {
          if (rows.hasNext) {
            throw new IOException("Expected single row in partition but received more.")
          }
          val metrics = Array.fill(watches.size)(new Array[Float](numRounds))
          val booster = XGBoost.train(
            watches(trainMatrix), params, numRounds, watches, metrics,
            earlyStoppingRound = earlyStoppingRound)
          val bytes = booster.toByteArray
          booster.dispose
          Iterator(bytes -> watches.keys.zip(metrics).toMap)
        } finally {
          watches.values.foreach(_.delete())
        }
      } finally {
        rabitEnv.foreach { _ => Rabit.shutdown() }
      }
    }).cache()
}

/**
  * Public entry point for xgboost4j training.
  */
object MlrXGBoost {
  private final val N_THREAD = "nthread"

  private def overrideParamsAccordingToTaskCPUs(sc: SparkContext, params: Map[String, Any]): Map[String, Any] = {
    val coresPerTask = sc.getConf.getInt("spark.task.cpus", 1)
    if (params.contains(N_THREAD)) {
      val nThread = params(N_THREAD).toString.toInt
      require(nThread <= coresPerTask,
        s"the nthread configuration ($nThread) must be no larger than " +
        s"spark.task.cpus ($coresPerTask)")
      params
    } else {
      params + (N_THREAD -> coresPerTask)
    }
  }

  private def startTracker(nWorkers: Int, trackerConf: TrackerConf): IRabitTracker = {
    val tracker: IRabitTracker = trackerConf.trackerImpl match {
      case "scala" => MjolnirUtils.scalaRabitTracker(nWorkers)
      case "python" => new PyRabitTracker(nWorkers)
      case _ => new PyRabitTracker(nWorkers)
    }

    require(tracker.start(trackerConf.workerConnectionTimeout), "FAULT: Failed to start tracker")
    tracker
  }

  private def postTrackerReturnProcessing(
    trackerReturnVal: Int,
    distributedBoosters: RDD[(Array[Byte], Map[String, Array[Float]])],
    trainMatrix: String,
    async: Option[FutureAction[_]]
  ): XGBoostModel = {
    if (trackerReturnVal != 0) {
      async.foreach(_.cancel())
      throw new Exception("XGBoostModel training failed")
    }
    val res = distributedBoosters.first()
    distributedBoosters.unpersist()
    val bais = new ByteArrayInputStream(res._1)
    val booster = XGBoost.loadModel(bais)
    MjolnirUtils.model(booster, res._2, trainMatrix)
  }

  /**
    * Primary entry point for training xgboost models. Fold data is typically
    * generated by the make_folds.py script.
    *
    * @param sc SparkContext to run training in
    * @param fold Partitioned training data with a map per partition
    * @param trainMatrix The name of the matrix in fold to train against
    * @param params XGBoost training parameters
    * @param numRounds The number of boosting rounds to perform
    * @param earlyStoppingRound Stop training after this many rounds of no improvement
    *                           to the test set. Set to 0 to disable.
    * @return Trained XGBoost model
    */
  def trainWithFiles(
    jsc: JavaSparkContext,
    fold: Seq[Map[String, String]],
    trainMatrix: String,
    params: Map[String, Any],
    numRounds: Int,
    earlyStoppingRound: Int
  ): XGBoostModel = {
    val sc = jsc.sc
    // Convert input data, a map of names to paths of worker splits, into
    // a map per worker containing it's individual splits.
    val baseData = fold.indices.zip(fold)
    // Distribute that data into one row per partition
    val rdd = sc.parallelize(baseData, baseData.length)
      .partitionBy(new ExactPartitioner(baseData.length, baseData.length))
      .map(_._2)

    val trainer = new MlrXGBoost(new AsLocalFile(sc))
    val overwrittenParams = overrideParamsAccordingToTaskCPUs(sc, params)
    if (rdd.getNumPartitions == 1) {
      // Special case with single worker, doesn't need Rabit
      val distributedBoosters = trainer.buildDistributedBoosters(
        rdd, trainMatrix, overwrittenParams, None, numRounds,
        earlyStoppingRound)
      distributedBoosters.foreachPartition(() => _)
      postTrackerReturnProcessing(0, distributedBoosters, trainMatrix, None)
    } else {
      val trackerConf = params.get("tracker_conf") match {
        case None => TrackerConf()
        case Some(conf: TrackerConf) => conf
        case _ => throw new IllegalArgumentException("parameter \"tracker_conf\" must be an " +
          "instance of TrackerConf.")
      }

      val tracker = startTracker(baseData.length, trackerConf)
      try {
        val distributedBoosters = trainer.buildDistributedBoosters(
          rdd, trainMatrix, overwrittenParams, Some(tracker.getWorkerEnvs),
          numRounds, earlyStoppingRound)
        val async = distributedBoosters.foreachPartitionAsync(() => _)
        postTrackerReturnProcessing(tracker.waitFor(0L), distributedBoosters, trainMatrix, Some(async))
      } finally {
        tracker.stop()
      }
    }
  }
}

// Necessary to ensure we distribute the training files as 1 file per partition
private class ExactPartitioner[V](partitions: Int, elements: Int) extends Partitioner {
  override def numPartitions: Int = partitions
  override def getPartition(key: Any): Int = {
    val k = key.asInstanceOf[Int]
    k * partitions / elements
  }
}


