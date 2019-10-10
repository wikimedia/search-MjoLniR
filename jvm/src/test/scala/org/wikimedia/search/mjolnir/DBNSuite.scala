package org.wikimedia.search.mjolnir

import org.apache.spark.sql.{Row, types => T}
import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.json4s.{JArray, JBool, JString}
import org.json4s.jackson.JsonMethods

import scala.io.Source
import scala.util.Random

class DBNSuite extends SharedSparkContext {
  test("create session items") {
    val ir = new InputReader(1, 20, true)
    val item = ir.makeSessionItem(
      "foo", "enwiki",
      Array("Example", ".example", "example.com"),
      Array(false, true, false))

    assert(item.isDefined)
  }

  test("session items are truncated to serpSize") {
    val serpSize = 20
    val ir = new InputReader(1, serpSize, true)
    val urls = (0 to 30).map(_.toString).toArray
    val clicks = Array.fill(30)(false)
    clicks(2) = true

    val maybeItem = ir.makeSessionItem("foo", "enwiki", urls, clicks)
    assert(maybeItem.isDefined)
    val item = maybeItem.get
    assert(item.clicks.length == serpSize)
    assert(item.urlIds.length == serpSize)
  }

  test("no urls gives no session item") {
    val ir = new InputReader(1, 2, true)
    val urls = new Array[String](0)
    val clicks = new Array[Boolean](0)
    assert(ir.makeSessionItem("foo", "enwiki", urls, clicks).isEmpty)
  }

  test("no clicks gives no session item") {
    val ir = new InputReader(1, 2, true)
    val urls = Array("a", "b", "c")
    val clicks = Array(false, false, false)
    assert(ir.makeSessionItem("foo", "enwiki", urls, clicks).isEmpty)
  }

  test("clicks are padded with false up to url count") {
    val ir = new InputReader(1, 5, true)
    val urls = (0 until 5).map(_.toString).toArray
    val clicks = Array(false, true)
    val maybeItem = ir.makeSessionItem("foo", "enwiki", urls, clicks)
    assert(maybeItem.isDefined)
    val item = maybeItem.get
    assert(item.clicks.length == 5)
    assert(item.clicks(1))
    assert(item.clicks.map(if (_) 1 else 0).sum == 1)
  }

  test("create session item from line") {
    val ir = new InputReader(1, 20, true)
    val line = s"""hash_digest${"\t"}foo${"\t"}enwiki${"\t"}intentWeight${"\t"}["Example", ".example", "example.com"]${"\t"}layout${"\t"}[false, true, false]"""

    val items = PyBridge.read(ir, Iterator(line))
    assert(items.length == 1)
    val item = items.head
    assert(item.clicks.length == 3)
    assert(item.urlIds.length == 3)
  }

  test("compare results vs python clickmodels") {
    val file = Source.fromURL(getClass.getResource("/dbn.data"))
    val ir = new InputReader(1, 20, true)
    val sessions = PyBridge.read(ir, file.getLines())
    val config = ir.config(0.5D, 1)
    val model = new DbnModel(0.9D, config)
    val urlRelevances = model.train(sessions)
    val relevances = ir.toRelevances(urlRelevances)

    assert(relevances.length == 8)

    val queries = relevances.map { x => x.query }.toSet
    assert(queries.size == 2)
    assert(queries.contains("12345"))
    assert(queries.contains("23456"))

    val regions = relevances.map { x => x.region }.toSet
    assert(regions.size == 1)
    assert(regions.contains("foowiki"))

    // Values sourced from python clickmodels implementation
    val expected = Map(
      ("23456", "1111") -> 0.1D,
      ("23456", "2222") -> 0.2322820037105751D,
      ("23456", "3333") -> 0.16054421768707483D,
      ("23456", "4444") -> 0.3424860853432282D,
      ("12345", "1111") -> 0.38878221072530095D,
      ("12345", "2222") -> 0.33396748936638354D,
      ("12345", "3333") -> 0.23195153695743548D,
      ("12345", "4444") -> 0.23523307569244717D)

    for( rel <- relevances) {
      assert(math.abs(expected((rel.query, rel.url)) - rel.relevance) < 0.0001D)
    }
  }

  test("providing more results than expected still works") {
    val N = 30
    val clicks = Array.fill(N)(false)
    clicks(2) = true

    val sessions = Seq(
      new SessionItem(0, (0 until N).toArray, clicks),
      new SessionItem(0, (0 until N).toArray, clicks)
    )

    val config = new Config(0, 0.5D, 2, 20, Array(N))
    val model = new DbnModel(0.9D, config)
    model.train(sessions)
    // no exceptions thrown
    assert(true)
  }

  test("backwards forwards") {
    val rel = new PositionRel(
      Array.fill(20)(0.5D), Array.fill(20)(0.5D)
    )
    val gamma = 0.9D
    val clicks = Array.fill(20)(false)

    val model = new DbnModel(0.5D, new Config(0, 0.5D, 1, 20, Array(20)))
    model.calcForwardBackwardEstimates(rel, clicks)
    val alpha = model.alpha
    val beta = model.beta
    val x = alpha(0)(0) * beta(0)(0) + alpha(0)(1) * beta(0)(1)

    val ok: Array[Boolean] = alpha.zip(beta).map { case (a: Array[Double], b: Array[Double]) =>
      math.abs((a(0) * b(0) + a(1) * b(1)) / x - 1) < 0.00001D
    }

    assert(ok.forall(x => x))
  }

  test("session estimate") {
    // Values sourced from python clickmodels implementation
    val rel = new PositionRel(Array.fill(20)(0.5D), Array.fill(20)(0.5D))
    val clicks = Array.fill(20)(false)
    val model = new DbnModel(0.9D, new Config(0, 0.5D, 1, 20, Array(20)))

    clicks(0) = true
    var sessionEstimate = model.getSessionEstimate(rel, clicks)
    assert(math.abs(sessionEstimate.a.sum - 10.4370D) < 0.0001D)
    assert(math.abs(sessionEstimate.s.sum - 0.8461D) < 0.0001D)
    assert(math.abs(sessionEstimate.s.sum - sessionEstimate.s(0)) < 0.0001D)

    clicks(10) = true
    sessionEstimate = model.getSessionEstimate(rel, clicks)
    assert(math.abs(sessionEstimate.a.sum - 6.4347D) < 0.0001D)
    assert(math.abs(sessionEstimate.s.sum - 0.8457D) < 0.0001D)
    assert(math.abs(sessionEstimate.s.sum - sessionEstimate.s(0) - sessionEstimate.s(10)) < 0.0001D)
  }

  private val FULL_BENCHMARK = false

  // Takes ~1.5s on my laptop versus 90 seconds in python
  test("basic benchmark") {
    val nQueries = if (FULL_BENCHMARK) 5000 else 100
    val nSessionsPerQuery = 20
    val nIterations = 40
    val nResultsPerQuery = 20

    val r = new Random(0)
    val ir = new InputReader(10, 20, true)
    val sessions = (0 until nQueries).flatMap { query =>
      val urls: Array[String] = (0 until nResultsPerQuery).map { _ => r.nextInt.toString }.toArray
      (0 until nSessionsPerQuery).flatMap { session =>
        val clicks = Array.fill(nResultsPerQuery)(false)
        do {
          clicks(r.nextInt(nResultsPerQuery)) = true
        } while(r.nextDouble > 0.95D)
        ir.makeSessionItem(query.toString, "region", urls, clicks)
      }
    }.toArray.toSeq


    assert(sessions.length == nQueries * nSessionsPerQuery)

    val config = ir.config(0.5D, nIterations)
    val dbn = new DbnModel(0.9D, config)
    (0 until 5).foreach { _ =>
      val start = System.nanoTime()
      dbn.train(sessions)
      val took = System.nanoTime() - start
      println(s"Took ${took / 1000000}ms")
    }

    // Create a datafile that python clickmodels can read in to have fair comparison
    //import java.io.File
    //import java.io.PrintWriter

    //val writer = new PrintWriter(new File("/tmp/dbn.clickmodels"))
    //for ( s <- sessions) {
    //  // poor mans json serialization
    //  val layout = Array.fill(s.urlIds.length)("false").mkString("[", ",", "]")
    //  val clicks = s.clicks.map(_.toString).mkString("[", ",", "]")
    //  val urls = s.urlIds.map(_.toString).mkString("[\"", "\",\"", "\"]")
    //  writer.write(s"0\t${s.queryId}\tregion\t0\t$urls\t$layout\t$clicks\n")
    //}
    //writer.close()
  }

  private val nextSessionId: () => String = {
    var current: Int = 0;
    { () =>
      current += 1
      current.toString
    }
  }

  private def makeSession(nHits: Integer): Seq[(Row, (Long, Int))] = {
    val sessionId = nextSessionId()
    val nQueries = Random.nextInt(2) + 1
    (0 until nQueries).flatMap { _ =>
      val normQueryId = Math.abs(Random.nextLong() % 10)
      (0 until nHits).map { k =>
        val pageId = Random.nextInt(100)
        // Guarantee sessions all have at least one click
        val clicked = if (k == 0) true else Random.nextFloat() * (k + 1) < 0.5
        val row = new GenericRow(Array(
          "testwiki", normQueryId, sessionId, k, pageId, clicked
        ))
        val pair = (normQueryId, pageId)
        (row, pair)
      }
    }
  }

  private val schema = T.StructType(
    T.StructField("wikiid", T.StringType) ::
    T.StructField("norm_query_id", T.LongType) ::
    T.StructField("session_id", T.StringType) ::
    T.StructField("hit_position", T.IntegerType) ::
    T.StructField("hit_page_id", T.IntegerType) ::
    T.StructField("clicked", T.BooleanType) ::
    Nil)

  test("train from a dataframe should not fail on simple query") {
    val (rows, _) = (makeSession(20) ++ makeSession(20)).unzip
    val df = spark.createDataFrame(spark.sparkContext.parallelize(rows), schema)
    DBN.train(df, Map()).collect()
  }

  test("empty partitions should not fail") {
    val rows: Seq[Row] = Seq()
    val df = spark.createDataFrame(spark.sparkContext.parallelize(rows), schema)
    DBN.train(df, Map()).collect()
  }

  test("multiple observations of the same wiki+query+page id should only be returned once") {
    val (oneSessionRows, pairs) = makeSession(20).unzip
    val rows = (0 to 5).flatMap { _ => oneSessionRows }
    val df = spark.createDataFrame(spark.sparkContext.parallelize(Random.shuffle(rows), 42), schema)
    val res = DBN.train(df, Map(
      "MAX_DOCS_PER_QUERY" -> "20"
    )).collect()
    val expected = pairs.groupBy(_._1).map { x => Math.min(x._2.toSet.size, 20) }.sum
    assert(res.length == expected)
  }

  // Helpers to compare python impl
  private object PyBridge {
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

    val PIECE_HASH_DIGEST = 0
    val PIECE_QUERY = 1
    val PIECE_REGION = 2
    val PIECE_INTENT_WEIGHT = 3
    val PIECE_URLS = 4
    val PIECE_LAYOUT = 5
    val PIECE_CLICKS = 6

    // TODO: Ideally dont use this and make session items directly without extra ser/deser overhead
    // This is primarily for compatability with the input format of python clickmodels library.
    def read(reader: InputReader, f: Iterator[String]): Seq[SessionItem] = {
      val sessions = f.flatMap { line =>
        val pieces = line.split("\t")
        val query: String = pieces(PIECE_QUERY)
        val region = pieces(PIECE_REGION)
        val urls = parseJsonStringArray(pieces(PIECE_URLS))
        val clicks = parseJsonBooleanArray(pieces(PIECE_CLICKS))

        reader.makeSessionItem(query, region, urls, clicks)
      }.toSeq
      // Guarantee we return a materialized collection and not a lazy one
      // which wont have properly updated our max query/url ids
      sessions.last
      sessions
    }
  }
}
