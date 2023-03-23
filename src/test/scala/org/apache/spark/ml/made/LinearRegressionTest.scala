package org.apache.spark.ml.made

import breeze.linalg.sum
import com.google.common.io.Files
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec._
import org.scalatest.matchers._

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.0000001
  lazy val data: DataFrame = LinearRegressionTest._data
  lazy val vectors = LinearRegressionTest._vectors

  "Model" should "scale input data" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      slope = Vectors.dense(2.0, -0.5).toDense,
      intercept = Vectors.dense(1.5, 0.5).toDense
    ).setInputCol("features")
      .setOutputCol("features")

    val vectors: Array[Vector] = model.transform(data).collect().map(_.getAs[Vector](0))

    vectors.length should be(2)

    validateModel(model, model.transform(data))
  }

  "Model" should "scale input data without mean subtraction" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      slope = Vectors.dense(1, 1).toDense,
      intercept = Vectors.dense(1).toDense
    ).setInputCol("features")
      .setOutputCol("features")

    val copy = model.copy(ParamMap(model.shiftMean -> false))

    val vectors: Array[Vector] = copy.transform(data).collect().map(_.getAs[Vector](0))

    vectors.length should be(2)

    vectors(0)(0) should be((13.5) / 1.5 +- delta)
    vectors(0)(1) should be((12) / 0.5 +- delta)

    vectors(1)(0) should be((-1) / 1.5 +- delta)
    vectors(1)(1) should be((0) / 0.5 +- delta)
  }

  "Estimator" should "calculate slope" in {
    val estimator = new LinearRegression()
      .setInputCol("features")
      .setOutputCol("features")

    val model = estimator.fit(data)

    model.slope(0) should be(1.0 +- delta)
    model.slope(1) should be(1.0 +- delta)
  }

  "Estimator" should "calculate intercept" in {
    val estimator = new LinearRegression()
      .setInputCol("features")
      .setOutputCol("features")

    val model = estimator.fit(data)

    model.intercept(0) should be(1.0 +- delta)
  }

  "Estimator" should "should produce functional model" in {
    val estimator = new LinearRegression()
      .setInputCol("features")
      .setOutputCol("features")

    val model = estimator.fit(data)

    validateModel(model, model.transform(data))
  }

  private def validateModel(model: LinearRegressionModel, data: DataFrame) = {
    val vectors: Array[Vector] = data.collect().map(_.getAs[Vector](0))

    vectors.length should be(2)

    vectors(0)(0) should be((13.5 - model.slope(0)) / model.intercept(0) +- delta)
    vectors(0)(1) should be((12 - model.slope(1)) / model.intercept(1) +- delta)

    vectors(1)(0) should be((-1 - model.slope(0)) / model.intercept(0) +- delta)
    vectors(1)(1) should be((0 - model.slope(1)) / model.intercept(1) +- delta)
  }

  "Estimator" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setOutputCol("features")
    ))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)

    val model = reRead.fit(data).stages(0).asInstanceOf[LinearRegressionModel]

    model.slope(0) should be(vectors.map(_(0)).sum / vectors.length +- delta)
    model.slope(1) should be(vectors.map(_(1)).sum / vectors.length +- delta)

    validateModel(model, model.transform(data))
  }

  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setOutputCol("features")
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
