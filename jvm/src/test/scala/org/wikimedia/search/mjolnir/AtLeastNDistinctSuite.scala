package org.wikimedia.search.mjolnir

import org.apache.spark.sql.expressions.MutableAggregationBuffer
import org.scalatest.FunSuite

class DummyBuffer(init: Array[Any]) extends MutableAggregationBuffer {
  val values: Array[Any] = init
  def update(i: Int, value: Any): Unit = values(i) = value
  def get(i: Int) = values(i)
  def length: Int = init.length
  def copy() = new DummyBuffer(init.clone())
}

class AtLeastNDistinctSuite extends FunSuite {
  import org.scalatest.prop.TableDrivenPropertyChecks._

  test("basic operation") {
    val udaf = new AtLeastNDistinct
    val buf = new DummyBuffer(new Array(udaf.bufferSchema.length))
    val row = new DummyBuffer(new Array(udaf.inputSchema.length))

    forAll(Table(
      ("limit", "expected", "values"),
      (1, false, Seq()),
      (1, true, Seq("zomg")),
      (1, true, Seq("hi", "hi", "hi")),
      (2, false, Seq("hi", "hi", "hi")),
      (2, true, Seq("hi", "there", "hi"))
    )) { (limit: Int, expect: Boolean, values: Seq[String]) =>
      udaf.initialize(buf)
      row(udaf.input_limit) = limit
      values.foreach { value =>
        row(udaf.input_value) = value
        udaf.update(buf, row)
      }
      assert(udaf.evaluate(buf) == expect)
    }
  }

  test("merge") {
    val udaf = new AtLeastNDistinct
    val buf1 = new DummyBuffer(new Array(udaf.bufferSchema.length))
    val buf2 = new DummyBuffer(new Array(udaf.bufferSchema.length))
    val row = new DummyBuffer(new Array(udaf.inputSchema.length))

    forAll(Table(
      ("limit", "expected", "a", "b"),
      (1, true, Set("a"), Set[String]()),
      (1, true, Set[String](), Set("a")),
      (2, false, Set("a"), Set("a")),
      (2, true, Set("a"), Set("b"))
    )) { (limit: Int, expect: Boolean, a: Set[String], b: Set[String]) =>
      udaf.initialize(buf1)
      udaf.initialize(buf2)
      row(udaf.input_limit) = limit
      a.foreach { value =>
        row(udaf.input_value) = value
        udaf.update(buf1, row)
      }
      b.foreach { value =>
        row(udaf.input_value) = value
        udaf.update(buf2, row)
      }

      udaf.merge(buf1, buf2)
      assert(udaf.evaluate(buf1) == expect)
    }
  }
}
