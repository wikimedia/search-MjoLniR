package org.wikimedia.search.mjolnir

import scala.collection.mutable
import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types.{BooleanType, DataType, IntegerType, ObjectType, StringType, StructField, StructType}

/**
  * Spark can't perform a count distinct with a window operation, which we need
  * to drastically simplify some code in the query normalization phase. As long as
  * we already have to implement a custom aggregation step, make it a bit more efficient
  * by looking for at least N distinct values, rather than doing a count of the complete
  * number of distinct values which could be in the hundreds or low thousands. This
  * should be much nicer to memory and cpu than a full distinct count with set's.
  *
  * The second parameter to the UDAF, the limit, must be a literal value or the
  * results of aggregation is undefined.
  */
class AtLeastNDistinct extends UserDefinedAggregateFunction {
  // This is the input type for the aggregate function
  override def inputSchema: StructType = StructType(
    StructField("value", StringType) ::
    StructField("limit", IntegerType)  :: Nil)

  // This is the internal fields we keep for computing the aggregate
  override def bufferSchema: StructType = StructType(
    StructField("set", ObjectType(mutable.Set.getClass)) ::
    StructField("limit", IntegerType) ::
    StructField("reached", BooleanType) ::  Nil)

  // Indexes into input and buffer schema to make code more readable
  final val input_value = 0
  final val input_limit = 1
  final val buffer_set = 0
  final val buffer_limit = 1
  final val buffer_reached = 2

  // This is the output type of the aggregation function
  override def dataType: DataType = BooleanType

  override def deterministic: Boolean = true

  // This is the initial value for the buffer schema
  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    // Wish we could initialize the limit here as well...
    buffer(buffer_set) = mutable.Set[String]()
    buffer(buffer_reached) = false
  }

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    // We have to store the limit, or it won't be available in the merge step
    buffer(buffer_limit) = input.getAs[Int](input_limit)
    if (!buffer.getAs[Boolean](buffer_reached)) {
      getSet(buffer) += input.getAs[String](input_value)
      checkReached(buffer)
    }
  }

  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    if (buffer2.getAs[Boolean](buffer_reached)) {
      buffer1(buffer_reached) = true
    } else if (!buffer1.getAs[Boolean](buffer_reached)) {
      getSet(buffer1) ++= getSet(buffer2)
      checkReached(buffer1)
    }
  }

  override def evaluate(buffer: Row): Any = buffer.getAs[Boolean](buffer_reached)

  private def getSet(buffer: Row): mutable.Set[String] =
    buffer.getAs[mutable.Set[String]](buffer_set)

  private def checkReached(buffer: MutableAggregationBuffer): Unit = {
    val set = getSet(buffer)
    val limit = buffer.getAs[Int](buffer_limit)
    if (set.size >= limit) {
      buffer(buffer_set) = set.empty
      buffer(buffer_reached) = true
    }
  }

}
