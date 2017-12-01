package org.wikimedia.search.mjolnir

/**
  * Implements a Dynamic Bayesian Network for click modeling. This is a probabilistic
  * graphical model used to predict relevance of (query, url) pairs from past
  * observations of user behaviour. Ported from the python clickmodels library.
  * Training uses probablistic EM (Expectation Maximization) method.
  *
  * A Dynamic Bayesian Network Click Model for Web Search Ranking - Olivier Chapelle and
  * Ya Zang - http://olivier.chapelle.cc/pub/DBN_www2009.pdf
  */
import scala.collection.mutable
import org.json4s.{JArray, JBool, JString}
import org.json4s.jackson.JsonMethods

case class SessionItem(queryId: Int, urlIds: Array[Int], clicks: Array[Boolean])
case class RelevanceResult(query: String, region: String, url: String, relevance: Double)

class InputReader(minDocsPerQuery: Int, serpSize: Int, discardNoClicks: Boolean) {

  // This bit maps input queryies/results to array indexes to be used while calculating
  private var currentUrlId: Int = 0 // TODO: Why is first returned value 1 instead of 0?
  private var currentQueryId: Int = -1
  private val urlToId: DefaultMap[String, Int] = new DefaultMap({ _ =>
    currentUrlId += 1
    currentUrlId
  })
  private val queryToId: DefaultMap[(String, String), Int] = new DefaultMap({ _ =>
    currentQueryId += 1
    currentQueryId
  })

  def maxQueryId: Int = currentQueryId + 2

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
    val n = math.min(serpSize, urls.length)
    val hasClicks = clicks.take(n).exists { x => x}
    if (urls.length < minDocsPerQuery ||
        (discardNoClicks && !hasClicks)
    ) {
      None
    } else {
      val queryId = queryToId((query, region))
      val urlIds = urls.take(n).map { url => urlToId(url) }
      Some(SessionItem(queryId, urlIds, clicks.take(n)))
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
  def read(f: Iterator[String]): Seq[SessionItem] = {
    f.flatMap { line =>
      val pieces = line.split("\t")
      val query: String = pieces(PIECE_QUERY)
      val region = pieces(PIECE_REGION)
      val urls = parseJsonStringArray(pieces(PIECE_URLS))
      val clicks = parseJsonBooleanArray(pieces(PIECE_CLICKS))

      makeSessionItem(query, region, urls, clicks)
    }.toSeq
  }

  def toRelevances(urlRelevances: Array[Map[Int, UrlRel]]): Seq[RelevanceResult] = {
    val idToUrl = urlToId.asMap.map(_.swap)
    val idToQuery = queryToId.asMap.map(_.swap)

    urlRelevances.zipWithIndex.flatMap { case (d, queryId) =>
      val (query, region) = idToQuery(queryId)
      d.map { case (urlId, urlRel) =>
        val url = idToUrl(urlId)
        RelevanceResult(query, region, url, urlRel.a * urlRel.s)
      }
    }
  }
}

class Config(val maxQueryId: Int, val defaultRel: Double, val maxIterations: Int)

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

case class SessionEstimate(
  a: (Double, Double), s: (Double, Double),
  e: Array[(Double, Double)], C: Double,
  clicks: Array[Double])


// Bit of a hack ... but to make things easy to deal with this makes
// it so requesting an item not in the map gets set to a default
// value and then returned. This differs from withDefault which
// expects to return an immutable value so doesn't set it into the map.
class DefaultMap[K, V](default: K => V) extends Iterable[(K, V)] {
  private val map = mutable.Map[K,V]()

  def apply(key: K): V = {
    map.get(key) match {
      case Some(value) => value
      case None =>
        val value = default(key)
        map.update(key, value)
        value
    }
  }

  override def iterator: Iterator[(K, V)] = map.iterator

  // converts to immutable scala Map
  def asMap: Map[K, V] = map.toMap
}

object DbnModel {
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
  def getForwardBackwardEstimates(rel: PositionRel, gamma: Double, clicks: Array[Boolean]): (Array[Array[Double]], Array[Array[Double]]) = {
    val N = clicks.length
    // alpha(i)(e) = P(C_1,...C_{i-1},E_i=e|a_u,s_u,G) calculated forwards for C_1, then C_1,C_2, ...
    val alpha = Array.ofDim[Double](N + 1, 2)
    // beta(i)(e) = P(C_{i+1},...C_N|E_i=e,a_u,s_u,G) calculated backwards for C_10, then C_9, C_10, ...
    val beta = Array.ofDim[Double](N + 1, 2)

    alpha(0)(1) = 1D
    beta(N)(0) = 1D
    beta(N)(1) = 1D

    // Forwards (alpha) and backwards (beta) need the same probabilities as inputs so pre-calculate them.
    var i = 0
    val updateMatrix: Array[Array[Array[Double]]] = Array.ofDim[Double](clicks.length, 2, 2)
    while (i < N) {
      val a_u = rel.a(i)
      val s_u = rel.s(i)
      if (clicks(i)) {
        updateMatrix(i)(0)(1) = (s_u + (1 - gamma) * (1 - s_u)) * a_u
        updateMatrix(i)(1)(1) = gamma * (1 - s_u) * a_u
      } else {
        updateMatrix(i)(0)(0) = 1D
        updateMatrix(i)(0)(1) = (1D - gamma) * (1D - a_u)
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
          alpha(i)(0) * updateMatrix(i)(1)(0) +
          alpha(i)(1) * updateMatrix(i)(1)(1)

        // beta(N-1-i)(e) = sum for e' in {0,1} of beta(N-1-i)(e') * updateMatrix(i)(e)(e')
        beta(N - 1 - i)(0) =
          beta(N - i)(0) * updateMatrix(N - 1 - i)(0)(0) +
          beta(N - i)(1) * updateMatrix(N - 1 - i)(1)(0)
        beta(N - 1 - i)(1) =
          beta(N - i)(0) * updateMatrix(N - 1 - i)(0)(1) +
          beta(N - i)(1) * updateMatrix(N - 1 - i)(1)(1)
      i += 1
    }

    (alpha, beta)
  }

  // Returns
  //  a: P(A_i|C_i,G) - Probability of attractiveness at position k conditioned on clicked and gamma
  //  s: P(S_i|C_i,G) - Probability of satisfaction at position k conditioned on clicked and gamma
  def getSessionEstimate(rel: PositionRel, gamma: Double, clicks: Array[Boolean]): PositionRel = {
    val N = clicks.length
    // alpha(i)(e) is P(C_1,...,C_{i-1},E_i=e|a_u,s_u,G)
    // beta(i)(e) is P(C_{i+1},...,C_N|E_i=e,a_u,s_u,G)
    val (alpha, beta) = DbnModel.getForwardBackwardEstimates(rel, gamma, clicks)

    // varphi is the smoothing of the forwards and backwards. I think, based on wiki page on forward/backwards
    // algorithm, that varphi is then P(E_i|C_1,...,C_N,a_u,s_u,G) but not 100% sure...
    var k = 0
    val varphi: Array[Double] = new Array(alpha.length)
    while (k < alpha.length) {
      val a = alpha(k)
      val b = beta(k)
      val ab0 = a(0) * b(0)
      val ab01 = ab0 + a(1) * b(1)
      varphi(k) = ab0 / ab01
      k += 1
    }

    val sessionEstimate = new PositionRel(new Array[Double](N), new Array[Double](N))
    k = 0
    while (k < N) {
      val a_u = rel.a(k)
      val s_u = rel.s(k)
      // E_i_multiplier --- P(S_i=0|C_i)P(C_i|E_i=1) (eq 6)
      if (clicks(k)) {
        // if user clicked attraction = 1 (eq 5a)
        sessionEstimate.a(k) = 1D
        // if user clicked satisfaction is (why?)
        //   prob of examination of next * satisfaction / (satisfaction + liklihood of abandonment * dissatisfaction)
        sessionEstimate.s(k) = varphi(k + 1) * s_u / (s_u + (1 - gamma) * (1 - s_u))
      } else {
        // with no click attraction = attraction * prob of examination (why?)
        sessionEstimate.a(k) = a_u * varphi(k)
        // with no click satisfaction = 0 (eq 5d)
        sessionEstimate.s(k) = 0D
      }
      k += 1
    }
    sessionEstimate
  }
}

class DbnModel(gamma: Double, config: Config) {

  def train(sessions: Seq[SessionItem]): Array[Map[Int, UrlRel]] = {
    // This is basically a multi-dimensional array with queryId in the first
    // dimension and urlId in the second dimension. Because queries only reference
    // a subset of the known urls we use a map at the second level instead of
    // creating the entire matrix.
    val urlRelevances: Array[DefaultMap[Int, UrlRel]] = Array.fill(config.maxQueryId) {
      new DefaultMap[Int, UrlRel]({
        _ => new UrlRel(config.defaultRel, config.defaultRel)
      })
    }

    for (_ <- 0 until config.maxIterations) {
      for ((d, queryId) <- eStep(urlRelevances, sessions).view.zipWithIndex) {
        // M step
        for ((urlId, relFractions) <- d) {
          val rel = urlRelevances(queryId)(urlId)
          // Convert our sums of per-session a_u and s_u into probabilities (domain of [0,1])
          // attracted / (attracted + not-attracted)
          rel.a = relFractions.a(1) / (relFractions.a(1) + relFractions.a(0))
          // satisfied / (satisfied + not-satisfied)
          rel.s = relFractions.s(1) / (relFractions.s(1) + relFractions.s(0))
        }
      }
    }

    urlRelevances.map(_.asMap)
  }

  // E step
  private def eStep(urlRelevances: Array[DefaultMap[Int, UrlRel]], sessions: Seq[SessionItem])
  : Array[DefaultMap[Int, UrlRelFrac]] = {
    // urlRelFraction(queryId)(urlId)
    val urlRelFractions: Array[DefaultMap[Int, UrlRelFrac]] = Array.fill(config.maxQueryId) {
      new DefaultMap[Int, UrlRelFrac]({
        _ => new UrlRelFrac(Array.fill(2)(1D), Array.fill(2)(1D))
      })
    }

    for (s <- sessions) {
      val positionRelevances = new PositionRel(
        s.urlIds.map(urlRelevances(s.queryId)(_).a),
        s.urlIds.map(urlRelevances(s.queryId)(_).s)
      )

      val sessionEstimate = DbnModel.getSessionEstimate(positionRelevances, gamma, s.clicks)
      for ((urlId, i) <- s.urlIds.view.zipWithIndex) {
        // update attraction
        val rel = urlRelFractions(s.queryId)(urlId)
        val estA = sessionEstimate.a(i)
        rel.a(0) += (1 - estA)
        rel.a(1) += estA
        if (s.clicks(i)) {
          // update satisfaction
          val estS = sessionEstimate.s(i)
          rel.s(0) += (1 - estS)
          rel.s(1) += estS
        }
      }
    }
    urlRelFractions
  }
}


