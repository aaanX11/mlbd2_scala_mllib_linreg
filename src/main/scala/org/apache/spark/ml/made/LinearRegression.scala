package org.apache.spark.ml.made

import breeze.linalg.Matrix.castOps
import breeze.linalg.norm
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasFitIntercept, HasInputCol, HasLabelCol, HasMaxIter, HasOutputCol, HasPredictionCol, HasStandardization, HasStepSize, HasTol, HasWeightCol}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}
import org.apache.spark.sql.functions.{col, lit}

import scala.annotation.tailrec


trait LinearRegressionParams extends HasMaxIter with HasTol
  with HasFitIntercept with HasFeaturesCol
  with HasLabelCol with HasPredictionCol with HasStepSize {
  def setFeaturesCol(value: String) : this.type = set(featuresCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setLabelCol(value: String): this.type = set(labelCol, value)

  def setFitIntercept(value: Boolean) : this.type = set(fitIntercept, value)

  def setTol(value: Double): this.type = set(tol, value)

  def setStepSize(value: Double): this.type = set(stepSize, value)

  def setMaxIter(value: Int): this.type = set(maxIter, value)


  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())
    SchemaUtils.checkColumnType(schema, getLabelCol, new VectorUDT())

    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getPredictionCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getLabelCol).copy(name = getPredictionCol))
    }
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("LinearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {

    // Used to convert untyped dataframes to datasets with vectors
    implicit val encoder : Encoder[Vector] = ExpressionEncoder()

    val listCols = List($(featuresCol), $(labelCol))
    //val listCols = List($(inputCol))
    //val vectors: Dataset[Vector] = dataset.select(listCols.map(m=>col(m)):_*).as[Vector]

    val allListCols = if ($(fitIntercept)) List($(featuresCol), "constCol", $(labelCol))  else listCols

    val vectors2: Dataset[Row] = dataset.select("*").withColumn("constCol", lit(1.0))

    val vectors4 = vectors2.select(allListCols.map(m=>col(m)):_*)
    val assembler = new VectorAssembler()
      .setInputCols(allListCols.toArray)
      .setOutputCol("all")

    val vectors3 = assembler.transform(vectors4)

    val vectors5 = vectors3.map(_.getAs[Vector]("all"))
    //vectors5.foreach(v => println(v))
    //val vectors3 = vectors2.map(row => row)

    //val labels: Dataset[Vector] = dataset.select(dataset("label").as[Vector])

    val nFeat: Int = AttributeGroup.fromStructField(dataset.schema($(featuresCol))).numAttributes.getOrElse(
      vectors5.first().size - 1
    )

    val weights0: Vector = Vectors.zeros(nFeat)

    val learnRate: Double = $(stepSize)

    val nRecord = vectors5.count() // ??? no shuffling ??

    @tailrec
    def descent(weights: Vector, iterCount: Int, grad0Norm: Double, gradNorm: Double): (Vector, Double) ={
      val diff = vectors5.rdd.map((x: Vector) =>
        (x.asBreeze(0 until nFeat).toDenseVector.dot(weights.asBreeze) - x.asBreeze(nFeat)) / nRecord.toDouble
      )
      if ((iterCount > 0) && (grad0Norm.isNaN || gradNorm / grad0Norm > $(tol))){


//        val whatisit = vectors5.rdd.map((x: Vector) =>
//          2.0 * learnRate * (x.asBreeze(0 until nFeat).toDenseVector.dot(weights.asBreeze) - x.asBreeze(nFeat)) * x.asBreeze(0 until nFeat).toVector
//        )

        val grad = (diff zip vectors5.rdd).map(t => t match {
          case (d, x) => 2.0 * d * x.asBreeze(0 until nFeat).toVector
        }).reduce((a, b) => a + b)
        val gradNormNew = norm(grad)
        val grad0NormNew = if (!grad0Norm.isNaN) grad0Norm else gradNormNew

        descent(Vectors.fromBreeze(weights.asBreeze - learnRate * grad), iterCount - 1, grad0NormNew, gradNormNew)
      } else {
        val loss = diff.reduce(_ + _)
        (weights, loss)
      }
    }

    val (weightsFound, loss) = descent(weights0, $(maxIter), Double.NaN, Double.NaN)

    if ($(fitIntercept)) {
      copyValues(new LinearRegressionModel(
        Vectors.fromBreeze(weightsFound.asBreeze(0 until nFeat - 1).toVector),
        Vectors.dense(weightsFound(nFeat - 1)))).setParent(this)
    } else {
      copyValues(new LinearRegressionModel(
        weightsFound,
        Vectors.dense(Double.NaN))).setParent(this)
    }
//    val Row(row: Row) =  dataset
//      .select(Summarizer.metrics("slope", "intercept").summary(dataset($(inputCol))))
//      .first()
//
//    copyValues(new LinearRegressionModel(row.getAs[Vector]("slope").toDense, row.getAs[Vector](1).toDense)).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                           override val uid: String,
                           val slope: DenseVector,
                           val intercept: DenseVector) extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {


  private[made] def this(slope: Vector, intercept: Vector) =
    this(Identifiable.randomUID("LinearRegressionModel"), slope.toDense, intercept.toDense)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(slope, intercept), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val bslope = slope.asBreeze
    val bintercept = intercept.asBreeze
    val transformUdf = if ($(fitIntercept)) {
      dataset.sqlContext.udf.register(uid + "_transform",
      (x : Vector) => {
        x.asBreeze.dot(bslope) + bintercept(0)
      })
    } else {
      dataset.sqlContext.udf.register(uid + "_transform",
        (x : Vector) => {
          (x.asBreeze.dot(bslope))
        })
    }

    dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors: (Vector, Vector) = slope.asInstanceOf[Vector] -> intercept.asInstanceOf[Vector]

      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder : Encoder[Vector] = ExpressionEncoder()

      val (slope, intercept) =  vectors.select(vectors("_1").as[Vector], vectors("_2").as[Vector]).first()

      val model = new LinearRegressionModel(slope, intercept)
      metadata.getAndSetParams(model)
      model
    }
  }
}
