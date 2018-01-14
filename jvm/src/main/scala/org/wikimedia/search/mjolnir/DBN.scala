package org.wikimedia.search.mjolnir

/**
  * Implements a Dynamic Bayesian Network for click modeling. This is a probabilistic
  * graphical model used to predict relevance of (query, url) pairs from past
  * observations of user behaviour. Ported from the python clickmodels library.
  * Training uses probablistic EM (Expectation Maximization) method.
  *
  * A Dynamic Bayesian Network Click Model for Web Search Ranking - Olivier Chapelle and
  * Ya Zang - http://olivier.chapelle.cc/pub/DBN_www2009.pdf
  *
  * It's worth noting that all of the math notes in this file are post-hoc. The
  * implementation was ported from python clickmodels by Aleksandr Chuklin and the
  * notes on math were added in an attempt to understand why the implementation works.
  */
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import org.apache.spark.sql.{DataFrame, Row, functions => F}
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.{types => T}
import org.json4s.{JArray, JBool, JString}
import org.json4s.jackson.JsonMethods

class SessionItem(val queryId: Int, val urlIds: Array[Int], val clicks: Array[Boolean])
class RelevanceResult(val query: String, val region: String, val url: String, val relevance: Double)

class InputReader(minDocsPerQuery: Int, maxDocsPerQuery: Int, discardNoClicks: Boolean) {

  // This bit maps input queries/results to array indexes to be used while calculating
  private val queryIdToNextUrlId: mutable.Map[Int, Int] = mutable.Map()
  private val queryIdToUrlToIdMap: mutable.Map[Int, mutable.Map[String, Int]] = mutable.Map()

  def urlToId(queryId: Int, url: String): Int = {
    val urlToIdMap = queryIdToUrlToIdMap.getOrElseUpdate(queryId, { mutable.Map() })
    urlToIdMap.getOrElseUpdate(url, {
      val nextUrlId = queryIdToNextUrlId.getOrElse(queryId, 0)
      queryIdToNextUrlId(queryId) = nextUrlId + 1
      nextUrlId
    })
  }

  private var nextQueryId: Int = 0
  private val queryToIdMap: mutable.Map[(String, String), Int] = mutable.Map()
  def queryToId(key: (String, String)): Int = {
    queryToIdMap.getOrElseUpdate(key, {
      nextQueryId += 1
      nextQueryId - 1
    })
  }

  private def parseJsonBooleanArray(json: String): Array[Boolean] = {
    JsonMethods.parse(json) match {
      case JArray(x: List[Any]) =>
        if (x.forall(_.isInstanceOf[JBool])) {
          x.asInstanceOf[List[JBool]].map(_.values).toArray
        } else {
          new Array[Boolean](0)
        }
      case _ => new Array[Boolean](0)
    }
  }

  private def parseJsonStringArray(json: String): Array[String] = {
    JsonMethods.parse(json) match {
      case JArray(x: List[Any]) =>
        if (x.forall(_.isInstanceOf[JString])) {
          x.asInstanceOf[List[JString]].map(_.values).toArray
        } else {
          new Array[String](0)
        }
      case _ => new Array[String](0)
    }
  }

  def makeSessionItem(query: String, region: String, urls: Array[String], clicks: Array[Boolean]): Option[SessionItem] = {
    val n = math.min(maxDocsPerQuery, urls.length)
    val allClicks: Array[Boolean] = if (clicks.length >= n) {
      clicks.take(n)
    } else {
      // pad clicks up to n with false
      val c: Array[Boolean] = Array.fill(n)(false)
      clicks.zipWithIndex.foreach { case (clicked, i) => c(i) = clicked }
      c
    }

    val hasClicks = allClicks.exists { x => x }
    if (urls.length < minDocsPerQuery ||
        (discardNoClicks && !hasClicks)
    ) {
      None
    } else {
      val queryId = queryToId((query, region))
      val urlIds = urls.take(n).map { url => urlToId(queryId, url) }
      Some(new SessionItem(queryId, urlIds, allClicks))
    }
  }

  val PIECE_HASH_DIGEST = 0
  val PIECE_QUERY = 1
  val PIECE_REGION = 2
  val PIECE_INTENT_WEIGHT = 3
  val PIECE_URLS = 4
  val PIECE_LAYOUT = 5
  val PIECE_CLICKS = 6

  // TODO: Ideally dont use this and make session items directly without extra ser/deser overhead
  // This is primarily for compatability with the input format of python clickmodels library.
  def read(f: Iterator[String]): Seq[SessionItem] = {
    val sessions = f.flatMap { line =>
      val pieces = line.split("\t")
      val query: String = pieces(PIECE_QUERY)
      val region = pieces(PIECE_REGION)
      val urls = parseJsonStringArray(pieces(PIECE_URLS))
      val clicks = parseJsonBooleanArray(pieces(PIECE_CLICKS))

      makeSessionItem(query, region, urls, clicks)
    }.toSeq
    // Guarantee we return a materialized collection and not a lazy one
    // which wont have properly updated our max query/url ids
    sessions.last
    sessions
  }

  def toRelevances(urlRelevances: Array[Array[UrlRel]]): Seq[RelevanceResult] = {
    val queryToUrlIdToUrl = queryIdToUrlToIdMap.map { case (queryId, urlToId) =>
      (queryId, urlToId.map(_.swap))
    }
    val queryIdToQuery = queryToIdMap.map(_.swap)

    urlRelevances.zipWithIndex.flatMap { case (d, queryId) =>
      val (query, region) = queryIdToQuery(queryId)
      val urlIdToUrl = queryToUrlIdToUrl(queryId)
      d.zipWithIndex.view.map { case (urlRel, urlId) =>
        val url = urlIdToUrl(urlId)
        new RelevanceResult(query, region, url, urlRel.a * urlRel.s)
      }
    }
  }


  def config(defaultRel: Double, maxIterations: Int): Config = {
    val maxUrlIds: Array[Int] = (0 until nextQueryId).map { queryId =>
      queryIdToNextUrlId(queryId) - 1
    }.toArray
    new Config(nextQueryId - 1, defaultRel, maxIterations, maxDocsPerQuery, maxUrlIds)
  }
}

class Config(val maxQueryId: Int, val defaultRel: Double, val maxIterations: Int, val maxDocsPerQuery: Int, val maxUrlIds: Array[Int]) {
  val maxUrlId: Int = maxUrlIds.max
}

// Some definitions:
//
// E_i = did the user examine the URL at position i
// A_i = was the user attracted by the url at position i
// S_i = was the user satisfied by the landing page at position i
// C_i = did the user click on position i
//
// (5a) A_i=1,E_i=1 <=> C_i = 1          - there is a click if and only if the user examined the url and was attracted
// (5b) a_u = P(A_i=1)                   - probability of being attracted depends only on the url
// (5c) s_u = P(S_i=1|C_i=1)             - when a user clicks a url there is a certain probability they will be satisfied
// (5d) C_i = 0 => S_i = 0               - if he user does not click they are not satisfied
// (5e) S_i = 1 => E_{i+1} = 0           - if the user is satisfied they stop examining pages
// (5f) P(E_{i+1}=1|E_i=1,S_i=0) = gamma - if the user is not satisfied by the current result there is a probability
//                                         (1 - gamma) that the user abandons their search and a probability gamma
//                                         that the user examines the next url
// (5g) E_i = 0 => E_{i+1} = 0           - If they did not examine position i, they will not examine later positions
//
// We also have:
//   r_u = P(S_i=1|E_i=1) - relevance of url u is the probability of satisfaction at position i conditioned on
//                          examination at position i
// Which can be decomposed into:
//   r_u = P(S_i=1|C_i=1)P(C_i=1|E_i=1) - relevance of url u is the probability of satisfaction at position i
//                                        conditioned on a click at position i multiplied by the probability of
//                                        click at position i conditioned on examination of position i
// Splitting those up we have:
//   s_u = P(S_i=1|C_i=1) - probability of satisfaction of url u at position i conditioned on click on position i.
//                          Also known as satisfaction, the ratio between actual and perceived relevance.
//   a_u = P(C_i=1|E_i=1) - probability of click of url u at position i conditioned on examination of position i.
//                          Also known as perceived relevance.

class UrlRel(var a: Double, var s: Double)
// Arrays in UrlRelFrac always have 2 items.
// First item: coefficient before \log r
// Second item: coefficient before \log (1 - r)
class UrlRelFrac(var a: Array[Double], var s: Array[Double])
// (same structure as above, but different name for clarity)
// attractiveness and satisfaction values for each position
class PositionRel(var a: Array[Double], var s: Array[Double])

class DbnModel(gamma: Double, config: Config) {
  val invGamma: Double = 1D - gamma

  def train(sessions: Seq[SessionItem]): Array[Array[UrlRel]] = {
    // This is basically a multi-dimensional array with queryId in the first
    // dimension and urlId in the second dimension. InputReader guarantees
    // that queryId starts at 0 and is continuous, and that per-query id urlId
    // also starts at 0 and is continuous, allowing static sized arrays to be used.
    val urlRelevances: Array[Array[UrlRel]] = (0 to config.maxQueryId).map { queryId =>
      (0 to config.maxUrlIds(queryId)).map { _ => new UrlRel(config.defaultRel, config.defaultRel) }.toArray
    }.toArray

    for (_ <- 0 until config.maxIterations) {
      val urlRelFractions = eStep(urlRelevances, sessions)
      var queryId = config.maxQueryId
      while (queryId >= 0) {
        // M step
        val queryUrlRelevances = urlRelevances(queryId)
        val queryUrlRelFractions = urlRelFractions(queryId)
        var urlId = config.maxUrlIds(queryId)
        // iterate over urls related to the query
        while (urlId >= 0) {
          val relFractions = queryUrlRelFractions(urlId)
          val rel = queryUrlRelevances(urlId)
          // Convert our sums of per-session a_u and s_u into probabilities (domain of [0,1])
          // attracted / (attracted + not-attracted)
          rel.a = relFractions.a(1) / (relFractions.a(1) + relFractions.a(0))
          // satisfied / (satisfied + not-satisfied)
          rel.s = relFractions.s(1) / (relFractions.s(1) + relFractions.s(0))

          // Reset rel-fractions for next iteration
          relFractions.a(0) = 1D
          relFractions.a(1) = 1D
          relFractions.s(0) = 1D
          relFractions.s(1) = 1D
          urlId -= 1
        }
        queryId -= 1
      }
    }

    urlRelevances
  }

  val positionRelevances = new PositionRel(new Array[Double](config.maxDocsPerQuery), new Array[Double](config.maxDocsPerQuery))
  // By pre-allocating we only have to fill the maps on the first iteration. After that we avoid
  // allocation and reuse what we already know we need. It's important that the train method reset these
  // to 1D after each iteration.
  //
  // urlRelFraction(queryId)(urlId)
  val urlRelFractions: Array[Array[UrlRelFrac]] = (0 to config.maxQueryId).map { queryId =>
    (0 to config.maxUrlIds(queryId)).map { _=>
      new UrlRelFrac(Array.fill(2)(1D), Array.fill(2)(1D))
    }.toArray
  }.toArray

  // E step
  private def eStep(urlRelevances: Array[Array[UrlRel]], sessions: Seq[SessionItem])
  : Array[Array[UrlRelFrac]] = {
    var sidx = 0
    while (sidx < sessions.length) {
      val s = sessions(sidx)
      var i = 0
      val urlRelQuery = urlRelevances(s.queryId)
      val N = Math.min(config.maxDocsPerQuery, s.urlIds.length)
      while (i < N) {
        val urlRel = urlRelQuery(s.urlIds(i))
        positionRelevances.a(i) = urlRel.a
        positionRelevances.s(i) = urlRel.s
        i += 1
      }

      val sessionEstimate = getSessionEstimate(positionRelevances, s.clicks)
      val queryUrlRelFrac = urlRelFractions(s.queryId)
      i = 0
      while (i < N) {
        val urlId = s.urlIds(i)
        // update attraction
        val rel = queryUrlRelFrac(urlId)
        val estA = sessionEstimate.a(i)
        rel.a(0) += 1 - estA
        rel.a(1) += estA
        if (s.clicks(i)) {
          // update satisfaction
          val estS = sessionEstimate.s(i)
          rel.s(0) += 1 - estS
          rel.s(1) += estS
        }
        i += 1
      }
      sidx += 1
    }
    urlRelFractions
  }

  // To keep from allocating while running the DBN create our intermediate
  // arrays at the largest size that might be needed. We must be careful to
  // never calculate based on the length of this, but instead of the lengths
  // of the input.
  val updateMatrix: Array[Array[Array[Double]]] = Array.ofDim(config.maxDocsPerQuery, 2, 2)
  // alpha(i)(e) = P(C_1,...C_{i-1},E_i=e|a_u,s_u,G) calculated forwards for C_1, then C_1,C_2, ...
  val alpha:Array[Array[Double]] = Array.ofDim(config.maxDocsPerQuery + 1, 2)
  // beta(i)(e) = P(C_{i+1},...C_N|E_i=e,a_u,s_u,G) calculated backwards for C_10, then C_9, C_10, ...
  val beta: Array[Array[Double]] = Array.ofDim(config.maxDocsPerQuery + 1, 2)
  val varphi: Array[Double] = new Array(config.maxDocsPerQuery + 1)

  /**
    * The forward-backward algorithm is used to to compute the posterior probabilities of the hidden variables.
    *
    * Derivation of update matrix. Equations from appendix of cited paper, E step:
    * updateMatrix(i)(e)(e') = P(E_{i+1}=e,C_i|E_i=e',G)
    *                        = sum for s in {0,1}: P(C_i=c|E_i=e')P(S_i=s|C_i=c)P(E_{i+1}=e|E_i=e',S_i=s)
    *
    * Derived update for the possible combinations of c, e and e':
    *
    * c=0, e=0, e'=0
    *   (1 * 1 * 1) + (1 * 0 * ?) = 1
    * c=0, e=0, e'=1
    *   ((1-a_u) * 1 * (1-gamma)) + ((1-a_u) * 0 * ?) = (1-a_u)*(1-gamma)
    * c=0, e=1, e'=0
    *   (1 * 1 * 0) + (1 * 0 * ?) = 0
    * c=0, e=1, e'=1
    *   ((1-a_u) * 1 * gamma) + ((1-a_u) * 0 * ?) = (1-a_u)*gamma
    * c=1, e=0, e'=0
    *   (0 * ? * ?) + (0 * ? * ?) = 0
    * c=1, e=0, e'=1
    *   (a_u * (1-s_u) * (1-gamma)) + (a_u * s_u * 1) = a_u * (((1-s_u) * (1-gamma)) + s_u)
    * c=1, e=1, e'=0
    *   (0 * ? * ?) + (0 * ? * ?) = 0
    * c=1, e=1, e'=1
    *   (a_u * (1-s_u) * gamma) + (a_u * s_u * 0) = a_u * (1-s_u) * gamma
    *
    * Above table derived from following identities:
    *
    * P(C_i=c|E_i=e')
    *
    * (5a) A_i=1,E_i=1 <=> C_i = 1
    * (6) P(C_i=1|E_i=1) = a_u
    *
    * P(C_i=1|E_i=1) = a_u   (6)
    * P(C_i=0|E_i=1) = 1 - a_u  (inverse of above)
    * P(C_i=1|E_i=0) = 0  (impossible per 5a)
    * P(C_i=0|E_i=0) = 1  (inverse of above)
    *
    * P(S_i=s|C_i=c)
    *
    * (5c) P(S_i=1|C_i=1) = s_u
    * (5d) C_i=0 => S_i=0
    *
    * P(S_i=0|C_i=1) = 1 - s_u (5c)
    * P(S_i=1|C_i=1) = s_u (inverse of above)
    * P(S_i=0|C_i=0) = 1 (5d)
    * P(S_i=1|C_i=0) = 0 (inverse of above)
    *
    * P(E_{i+1}=e|E_i=e',S_i=s)
    *
    * (5e) S_i = 1 => E_{i+1} = 0
    * (5f) P(E_{i+1}=1|E_i=1,S_i=0) = gamma
    * (5g) E_i = 0 => E_{i+1} = 0

    * P(E_{i+1}=0|E_i=1,S_i=0) = 1 - gamma (inverse of below)
    * P(E_{i+1}=1|E_i=1,S_i=0) = gamma (5f)
    * P(E_{i+1}=0|E_i=0,S_i=0) = 1 (5g)
    * P(E_{i+1}=1|E_i=0,S_i=0) = 0 (5g)
    * P(E_{i+1}=0|E_i=1,S_i=1) = 1 (5e)
    * P(E_{i+1}=1|E_i=1,S_i=1) = 0 (inverse of above)
    * P(E_{i+1}=0|E_i=0,S_i=1) = 1 (5g)
    * P(E_{i+1}=1|E_i=0,S_i=1) = 0 (5g)
    */
  def calcForwardBackwardEstimates(rel: PositionRel, clicks: Array[Boolean]): Unit = {
    val N = Math.min(config.maxDocsPerQuery, clicks.length)

    //always 0: alpha(0)(0) = 0D
    alpha(0)(1) = 1D
    beta(N)(0) = 1D
    beta(N)(1) = 1D

    // Forwards (alpha) and backwards (beta) need the same probabilities as inputs so pre-calculate them.
    var i = 0
    while (i < N) {
      val a_u = rel.a(i)
      val s_u = rel.s(i)
      if (clicks(i)) {
        updateMatrix(i)(0)(0) = 0D
        updateMatrix(i)(0)(1) = (s_u + invGamma * (1 - s_u)) * a_u
        // always 0: updateMatrix(i)(1)(0) = 0D
        updateMatrix(i)(1)(1) = gamma * (1 - s_u) * a_u
      } else {
        updateMatrix(i)(0)(0) = 1D
        updateMatrix(i)(0)(1) = invGamma * (1D - a_u)
        // always 0: updateMatrix(i)(1)(0) = 0D
        updateMatrix(i)(1)(1) = gamma * (1D - a_u)
      }
      i += 1
    }

    i = 0
    while (i < N) {
      // alpha(i+1)(e) = sum for e' in {0,1} of alpha(i)(e') * updateMatrix(i)(e)(e')
      alpha(i + 1)(0) =
        alpha(i)(0) * updateMatrix(i)(0)(0) +
        alpha(i)(1) * updateMatrix(i)(0)(1)
      alpha(i + 1)(1) =
        // always 0: alpha(i)(0) * updateMatrix(i)(1)(0) +
        alpha(i)(1) * updateMatrix(i)(1)(1)

      // beta(N-1-i)(e) = sum for e' in {0,1} of beta(N-1-i)(e') * updateMatrix(i)(e)(e')
      beta(N - 1 - i)(0) =
        beta(N - i)(0) * updateMatrix(N - 1 - i)(0)(0)
        // always 0: + beta(N - i)(1) * updateMatrix(N - 1 - i)(1)(0)
      beta(N - 1 - i)(1) =
        beta(N - i)(0) * updateMatrix(N - 1 - i)(0)(1) +
        beta(N - i)(1) * updateMatrix(N - 1 - i)(1)(1)
      i += 1
    }

    // (alpha, beta)
  }

  val sessionEstimate = new PositionRel(new Array[Double](config.maxDocsPerQuery), new Array[Double](config.maxDocsPerQuery))
  // Returns
  //  a: P(A_i|C_i,G) - Probability of attractiveness at position i conditioned on clicked and gamma
  //  s: P(S_i|C_i,G) - Probability of satisfaction at position i conditioned on clicked and gamma
  def getSessionEstimate(rel: PositionRel, clicks: Array[Boolean]): PositionRel = {
    val N = Math.min(config.maxDocsPerQuery, clicks.length)

    // This sets the instance variables alpha/beta
    // alpha(i)(e) is P(C_1,...,C_{k-1},E_i=e|a_u,s_u,G)
    // beta(i)(e) is P(C_{i+1},...,C_N|E_i=e,a_u,s_u,G)
    calcForwardBackwardEstimates(rel, clicks)

    // varphi is the smoothing of the forwards and backwards. I think, based on wiki page on forward/backwards
    // algorithm, that varphi is then P(E_i|C_1,...,C_N,a_u,s_u,G) but not 100% sure...
    var i = 0
    while (i < N + 1) {
      val a = alpha(i)
      val b = beta(i)
      val ab0 = a(0) * b(0)
      val ab01 = ab0 + a(1) * b(1)
      varphi(i) = ab0 / ab01
      i += 1
    }

    i = 0
    while (i < N) {
      val a_u = rel.a(i)
      val s_u = rel.s(i)

      // TODO: Clickmodels had this as S_i=0, but i'm pretty sure it's supposed to be =1
      // based on the actual updates performed?
      // E_i_multiplier --- P(S_i=0|C_i)P(C_i|E_i=1) (inverse of eq 6)
      if (clicks(i)) {
        // if user clicked attraction = 1 (eq 5a)
        sessionEstimate.a(i) = 1D
        // if user clicked satisfaction is (why?)
        //   prob of examination of next * satisfaction / (satisfaction + liklihood of abandonment * dissatisfaction)
        sessionEstimate.s(i) = varphi(i + 1) * s_u / (s_u + invGamma * (1 - s_u))
      } else {
        // probability of no click when examined = attraction * prob of examination (why?)
        sessionEstimate.a(i) = a_u * varphi(i)
        // with no click satisfaction = 0 (eq 5d)
        sessionEstimate.s(i) = 0D
      }
      i += 1
    }

    sessionEstimate
  }
}

private class DbnHitPage(val hitPageId: Int, val hitPosition: Double, val clicked: Boolean)

/**
  * Predict relevance of query/page pairs from individual user search sessions.
  */
object DBN {
  // TODO: These should all be configurable? Perhaps
  // also simplified somehow...
  private val CLICKED = "clicked"
  private val HITS = "hits"
  private val HIT_PAGE_ID = "hit_page_id"
  private val HIT_POSITION = "hit_position"
  private val NORM_QUERY_ID = "norm_query_id"
  private val RELEVANCE = "relevance"
  private val SESSION_ID = "session_id"
  private val WIKI_ID = "wikiid"

  /**
    * Given a sequence of rows representing multiple searches
    * for a single normalized query from a single session aggregate
    * hits into their average position and tag if it was clicked or not
    *
    * @param sessionHits Sequence of rows representing searches
    *                    for a single normalized query and session.
    * @return
    */
  private def deduplicateHits(sessionHits: Seq[Row]): (Array[String], Array[Boolean]) = {
    val deduped = sessionHits.groupBy(_.getAs[Int](HIT_PAGE_ID))
      .map { case (hitPageId, hits) =>
        val hitPositions = hits.map(_.getAs[Int](HIT_POSITION))
        val clicked = hits.exists(_.getAs[Boolean](CLICKED))
        val avgHitPosition = hitPositions.sum.toDouble / hitPositions.length.toDouble
        new DbnHitPage(hitPageId, avgHitPosition, clicked)
      }
      .toSeq.sortBy(_.hitPosition)
    val urls = deduped.map(_.hitPageId.toString).toArray
    val clicked = deduped.map(_.clicked).toArray
    (urls, clicked)
  }

  val trainOutputSchema = T.StructType(
    T.StructField(WIKI_ID, T.StringType) ::
    T.StructField(NORM_QUERY_ID, T.LongType) ::
    T.StructField(HIT_PAGE_ID, T.IntegerType) ::
    T.StructField(RELEVANCE, T.DoubleType) :: Nil)

  def train(df: DataFrame, dbnConfig: Map[String, String]): DataFrame = {
    val minDocsPerQuery = dbnConfig.getOrElse("MIN_DOCS_PER_QUERY", "10").toInt
    val maxDocsPerQuery = dbnConfig.getOrElse("MAX_DOCS_PER_QUERY", "10").toInt
    val defaultRel = dbnConfig.getOrElse("DEFAULT_REL", "0.9").toFloat
    val maxIterations = dbnConfig.getOrElse("MAX_ITERATIONS", "40").toInt
    val gamma = dbnConfig.getOrElse("GAMMA", "0.9").toFloat

    val dfGrouped = df
      // norm query id comes from monotonicallyIncreasingId, and as such is 64bit
      .withColumn(NORM_QUERY_ID, F.col(NORM_QUERY_ID).cast(T.LongType))
      .withColumn(HIT_PAGE_ID, F.col(HIT_PAGE_ID).cast(T.IntegerType))
      .withColumn(HIT_POSITION, F.col(HIT_POSITION).cast(T.IntegerType))
      .groupBy(WIKI_ID, NORM_QUERY_ID, SESSION_ID)
      .agg(F.collect_list(F.struct(HIT_POSITION, HIT_PAGE_ID, CLICKED)).alias(HITS))
      .repartition(F.col(WIKI_ID), F.col(NORM_QUERY_ID))

    val hitsIndex = dfGrouped.schema.fieldIndex(HITS)
    val normQueryIndex = dfGrouped.schema.fieldIndex(NORM_QUERY_ID)
    val wikiidIndex = dfGrouped.schema.fieldIndex(WIKI_ID)

    val rdd: RDD[Row] = dfGrouped
      .rdd.mapPartitions { rows: Iterator[Row] =>
        val reader = new InputReader(minDocsPerQuery, maxDocsPerQuery, discardNoClicks = true)
        val items = rows.flatMap { row =>
          // Sorts lowest to highest
          val (urls, clicked) = deduplicateHits(row.getSeq[Row](hitsIndex))
          val query = row.getLong(normQueryIndex).toString
          val region = row.getString(wikiidIndex)
          reader.makeSessionItem(query, region, urls, clicked)
        }.toSeq
        if (items.isEmpty) {
          Iterator()
        } else {
          // When we get a lazy seq from the iterator ensure its materialized
          // before creating config with the mutable state.
          items.length
          val config = reader.config(defaultRel, maxIterations)
          val model = new DbnModel(gamma, config)
          reader.toRelevances(model.train(items)).map { rel =>
            new GenericRowWithSchema(Array(rel.region, rel.query.toLong, rel.url.toInt, rel.relevance), trainOutputSchema)
          }.toIterator
        }
      }

    df.sqlContext.createDataFrame(rdd, trainOutputSchema)
  }
}
