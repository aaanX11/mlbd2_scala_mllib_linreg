package org.apache.spark.ml.made

import org.scalatest._
import org.scalatest.flatspec._
import org.scalatest.matchers._

//@Ignore
class StartSparkTest extends AnyFlatSpec with should.Matchers with WithSpark {

  "Spark" should "start context" in {
    System.out.println("Spark will die!")
    val s = spark
    System.out.println("Spark is!", s)
    System.out.println("2+2!")
    Thread.sleep(60000)
  }

}
