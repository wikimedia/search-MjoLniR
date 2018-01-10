package org.wikimedia.search.mjolnir

import java.io.{IOException, ObjectInputStream, ObjectOutputStream}
import java.nio.file.{Files, Path}

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path => HDFSPath}
import org.apache.spark.{SparkContext, TaskContext}
import org.apache.spark.broadcast.Broadcast

import scala.util.control.NonFatal

/**
  * Helper that makes hdfs paths appear local so xgboost can read them
  *
  * @param broadcastConfiguration The hadoop configuration to be used on executors
  *                               to access HDFS.
  */
class AsLocalFile(broadcastConfiguration: Broadcast[SerializableConfiguration]) extends Serializable {

  def this(sc: SparkContext) = this(sc.broadcast(new SerializableConfiguration(sc.hadoopConfiguration)))

  // Re-interpret files starting at root as local files
  private def asHDFSPath(path: String): HDFSPath = if (path.charAt(0) == '/') {
    new HDFSPath(s"file://$path")
  } else {
    new HDFSPath(path)
  }

  private def copyToLocalFile(src: String, dst: Path): Unit = {
    val s = asHDFSPath(src)
    val d = asHDFSPath(dst.toString)
    s.getFileSystem(broadcastConfiguration.value.value).copyToLocalFile(s, d)
  }

  /**
   * Convert string representing either a remote or local file into
   * a string representing a local file. If the file was copied locally
   * delete it on function exit after executing provided block.
   */
  def apply[A](path: String)(block: String => A): A = {
    if (path.startsWith("/")) {
      block(path)
    } else if (path.startsWith("file:/")) {
      block(path.substring("file:".length))
    } else {
      val prefix = s"mjolnir-${TaskContext.get.stageId()}-${TaskContext.getPartitionId()}-"
      val localOutputPath = Files.createTempFile(prefix, ".xgb")
      localOutputPath.toFile.deleteOnExit()
      try {
        copyToLocalFile(path, localOutputPath)
        block(localOutputPath.toString)
      } finally {
        Files.deleteIfExists(localOutputPath)
      }
    }
  }
}

/**
  * Makes hadoop configuration serializable as a broadcast variable
  */
class SerializableConfiguration(@transient var value: Configuration) extends Serializable {
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
