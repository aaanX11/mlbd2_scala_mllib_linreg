package org.apache.spark.ml.made

import breeze.linalg.{DenseVector, sum}
import com.google.common.io.Files
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Encoder, Row, functions}
import org.scalatest.Ignore
import org.scalatest.flatspec._
import org.scalatest.matchers._

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.001
  lazy val data: DataFrame = LinearRegressionTest._data
  lazy val vectors = LinearRegressionTest._vectors

  lazy val data0: DataFrame = LinearRegressionTest._data0
  lazy val vectors0 = LinearRegressionTest._vectors0

  lazy val dataRandom: DataFrame = LinearRegressionTest._dataRandom

  "Model" should "predict" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      slope = Vectors.dense(1.0, 1.0).toDense,
      intercept = Vectors.dense(1.5).toDense
    ).setFeaturesCol("features")
      .setLabelCol("label")
      .setStepSize(0.001)
      .setMaxIter(1000)
      .setTol(0.0001)

    //val vectors: Array[Vector] = model.transform(data).collect().map(_.getAs[Vector](0))
    val vectors: Array[Double] = model.transform(data).collect().map(_.getAs[Double](2))


    vectors.length should be(data.count())

    validateModel(model, model.transform(data))
  }

  "Estimator" should "calculate slope without intercept" in {
    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setStepSize(0.0002)
      .setMaxIter(2500)
      .setFitIntercept(false)
      .setTol(0.000001)

    val model = estimator.fit(data0)

    model.slope(0) should be(2.0 +- delta)
    model.slope(1) should be(1.0 +- delta)

    model.intercept(0).isNaN should be(true)
  }

  "Estimator" should "calculate slope" in {
    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setStepSize(0.002)
      .setMaxIter(2000)
      .setTol(0.000001)

    val model = estimator.fit(data)

    model.slope(0) should be(1.0 +- delta)
    model.slope(1) should be(1.0 +- delta)
  }

  "Estimator" should "fit random data" in {
    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setStepSize(0.04)
      .setMaxIter(5000)
      .setTol(1.0e-16)

    val model = estimator.fit(dataRandom)

    model.slope(0) should be(10.0 * 1.4142 +- delta)
    model.slope(1) should be(10.0 * 3.1415 +- delta)

    model.intercept(0) should be(42.0 +- delta)
  }


  "Estimator" should "calculate intercept" in {
    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setStepSize(0.005)
      .setMaxIter(2500)
      .setTol(0.00001)

    val model = estimator.fit(data)

    model.intercept(0) should be(1.0 +- delta)
  }

  "Estimator" should "should produce functional model" in {
    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setStepSize(0.005)
      .setMaxIter(2000)
      .setTol(0.00001)

    val model = estimator.fit(data)

    validateModel(model, model.transform(data))
  }

  private def validateModel(model: LinearRegressionModel, data: DataFrame) = {
    val vectors: Array[Double] = data.collect().map(_.getAs[Double](2))

    vectors.length should be(5)

    vectors(0) should be(0.0 * model.slope(0) + 12.0 * model.slope(1) + model.intercept(0) +- delta)
    vectors(1) should be(-1.0 * model.slope(0) + 0.0 * model.slope(1) + model.intercept(0) +- delta)
    vectors(2) should be(0.0 * model.slope(0) + 0.0 * model.slope(1) + model.intercept(0) +- delta)
    vectors(3) should be(-0.1 * model.slope(0) + -0.01 * model.slope(1) + model.intercept(0) +- delta)
    vectors(4) should be(5.0 * model.slope(0) + 4.0 * model.slope(1) + model.intercept(0) +- delta)

    model.slope(0) should be(1.0 +- delta)
    model.slope(1) should be(1.0 +- delta)
  }

  // not working on windows.
  ignore should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setStepSize(0.001)
        .setMaxIter(1500)
        .setTol(0.001)
    ))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)

    val model = reRead.fit(data).stages(0).asInstanceOf[LinearRegressionModel]

    //model.slope(0) should be(vectors.map(_(0)).sum / vectors.length +- delta)
    //model.slope(1) should be(vectors.map(_(1)).sum / vectors.length +- delta)

    validateModel(model, model.transform(data))
  }

  // not working on windows. Write part ok, read part fails with the error:
  // An exception or error caused a run to abort: 'org.apache.hadoop.io.nativeio.NativeIO$POSIX$Stat
  // org.apache.hadoop.io.nativeio.NativeIO$POSIX.stat(java.lang.String)'
  // java.lang.UnsatisfiedLinkError
  ignore should "work after re-read1" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setStepSize(0.001)
        .setMaxIter(1500)
        .setTol(0.001)
    ))

    val model = pipeline.fit(data)

    val tmpFolder = Files.createTempDir()
    val tmpFile = tmpFolder.getAbsolutePath
    println(tmpFile)
    model.write.overwrite().save(tmpFile)

    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)

    val dataTransformed = reRead.transform(data)
    validateModel(model.stages(0).asInstanceOf[LinearRegressionModel], dataTransformed)
  }
}

object LinearRegressionTest extends WithSpark {

  lazy val _vectors = Seq(
    Tuple2(Vectors.dense(0.0, 12.0), Vectors.dense(13.0)),
    Tuple2(Vectors.dense(-1.0, 0.0), Vectors.dense(0.0)),
    Tuple2(Vectors.dense(0.0, 0.0), Vectors.dense(1.0)),
    Tuple2(Vectors.dense(-0.1, -0.01), Vectors.dense(0.89)),
    Tuple2(Vectors.dense(5.0, 4.0), Vectors.dense(10.0))
  )

  lazy val _data: DataFrame = {
    import sqlc.implicits._
    _vectors.toDF("features", "label")
  }

  lazy val _vectors0 = Seq(
    Tuple2(Vectors.dense(100.0, 12.0), Vectors.dense(212.0)),
    Tuple2(Vectors.dense(45.0, 10.0), Vectors.dense(100.0)),
    Tuple2(Vectors.dense(0.25, 0.0), Vectors.dense(0.5)),
    Tuple2(Vectors.dense(-0.1, -0.01), Vectors.dense(-0.21)),
    Tuple2(Vectors.dense(-2.0, 4.0), Vectors.dense(0.0))
  )

  lazy val _data0: DataFrame = {
    import sqlc.implicits._
    _vectors0.toDF("features", "label")
  }

  val n = 100000
  val uni = breeze.stats.distributions.Uniform(-5.0, 5.0)
  val norm = breeze.stats.distributions.Gaussian(0.0, 0.6)
  val test1 = Seq.fill(n)(DenseVector.rand[Double](2, uni))
  val noise = Seq.fill(n)(norm.sample())

  val w = Vectors.dense(10.0 * 1.4142, 10.0 * 3.1415)
  val w0 = 42
  val _dataRandom: DataFrame = {
    import sqlc.implicits._
    test1.zip(noise).map(t => t match {
      case(x, n) => Tuple2(Vectors.fromBreeze(x), Vectors.dense(x.dot(w.asBreeze) + w0 + n))
    } ).toDF("features", "label")
    //test1.map(x => Tuple2(Vectors.fromBreeze(x), Vectors.dense(breeze.linalg.sum(x)))).toDF("features", "label")
  }

  //val test = _dataRandom.head(4)
  //val d = 1
}
