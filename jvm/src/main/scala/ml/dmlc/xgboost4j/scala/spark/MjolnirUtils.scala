package ml.dmlc.xgboost4j.scala.spark

import ml.dmlc.xgboost4j.java.IRabitTracker
import ml.dmlc.xgboost4j.scala.Booster
import ml.dmlc.xgboost4j.scala.rabit.RabitTracker

/**
  * Provide access to package-private constructs of xgboost4j-spark
  */
object MjolnirUtils {
  def model(booster: Booster, metrics: Map[String, Array[Float]], trainMatrix: String): XGBoostModel = {
    // Arbitrarily take an 'other' matrix if available
    val xgMetrics = metrics.keys.find(!_.equals(trainMatrix)).map{ name => Map(
      "train" -> metrics(trainMatrix),
      "test" -> metrics(name)
    ) }.getOrElse(Map(
      "train" -> metrics(trainMatrix)
    ))

    val model = new XGBoostRegressionModel(booster)
    model.setSummary(XGBoostTrainingSummary(xgMetrics))
    model
  }

  def scalaRabitTracker(nWorkers: Int): IRabitTracker = {
    new RabitTracker(nWorkers)
  }
}
