package org.apache.spark.ml.made

import breeze.linalg.{DenseVector, sum}
import com.google.common.io.Files
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.{ParamMap, ParamPair}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Row}
import org.scalatest.Ignore
import org.scalatest.flatspec._
import org.scalatest.matchers._

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.001
  lazy val data: DataFrame = LinearRegressionTest._data
  lazy val vectors = LinearRegressionTest._vectors



  "Model" should "predict" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      slope = Vectors.dense(1.0, 1.0).toDense,
      intercept = Vectors.dense(1.5).toDense
    ).setFeaturesCol("features")
      .setLabelCol("label")
      .setStepSize(0.001)
      .setMaxIter(1000)

    //val vectors: Array[Vector] = model.transform(data).collect().map(_.getAs[Vector](0))
    val vectors: Array[Double] = model.transform(data).collect().map(_.getAs[Double](2))


    vectors.length should be(data.count())

    validateModel(model, model.transform(data))
  }

  ignore should "calculate slope without intercept" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      slope = Vectors.dense(1, 1).toDense,
      intercept = Vectors.dense(1).toDense
    ).setFeaturesCol("features")
      .setLabelCol("label")
      .setStepSize(0.001)
      .setMaxIter(1000)

    //val copy = model.copy()

    val vectors: Array[Vector] = model.transform(data).collect().map(_.getAs[Vector](0))

    vectors.length should be(2)

    vectors(0)(0) should be((13.5) / 1.5 +- delta)
    vectors(0)(1) should be((12) / 0.5 +- delta)

    vectors(1)(0) should be((-1) / 1.5 +- delta)
    vectors(1)(1) should be((0) / 0.5 +- delta)
  }

  "Estimator" should "calculate slope" in {
    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setStepSize(0.001)
      .setMaxIter(1000)

    val model = estimator.fit(data)

    model.slope(0) should be(1.0 +- delta)
    model.slope(1) should be(1.0 +- delta)
  }

  "Estimator" should "calculate intercept" in {
    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setStepSize(0.001)
      .setMaxIter(1500)

    val model = estimator.fit(data)

    model.intercept(0) should be(1.0 +- delta)
  }

  "Estimator" should "should produce functional model" in {
    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setStepSize(0.001)
      .setMaxIter(1500)

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

  ignore should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setStepSize(0.001)
        .setMaxIter(1500)
    ))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)

    val model = reRead.fit(data).stages(0).asInstanceOf[LinearRegressionModel]

    model.slope(0) should be(vectors.map(_(0)).sum / vectors.length +- delta)
    model.slope(1) should be(vectors.map(_(1)).sum / vectors.length +- delta)

    validateModel(model, model.transform(data))
  }

  ignore should "work after re-read1" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setStepSize(0.001)
        .setMaxIter(1500)
    ))

    val model = pipeline.fit(data)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)

    validateModel(model.stages(0).asInstanceOf[LinearRegressionModel], reRead.transform(data))
  }
}

object LinearRegressionTest extends WithSpark {

  lazy val _vectors = Seq(
    Vectors.dense(0.0, 12.0, 13.0),
    Vectors.dense(-1.0, 0.0, 1.0),
    Vectors.dense(5.0, 4.0, 10.0)
  )

  lazy val _vectors2 = Seq(
    Tuple2(Vectors.dense(0.0, 12.0), Vectors.dense(13.0)),
    Tuple2(Vectors.dense(-1.0, 0.0), Vectors.dense(0.0)),
    Tuple2(Vectors.dense(0.0, 0.0), Vectors.dense(1.0)),
    Tuple2(Vectors.dense(-0.1, -0.01), Vectors.dense(0.89)),
    Tuple2(Vectors.dense(5.0, 4.0), Vectors.dense(10.0))
  )

  lazy val _data: DataFrame = {
    import sqlc.implicits._
    //_vectors.map(x => Tuple1(x)).toDF("features", "label")
    _vectors2.toDF("features", "label")
  }

}
