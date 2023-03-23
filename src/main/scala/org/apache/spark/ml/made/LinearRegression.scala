package org.apache.spark.ml.made

import breeze.linalg.Matrix.castOps
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.sql.functions.{col, lit}


trait LinearRegressionParams extends HasInputCol with HasOutputCol {
  def setInputCol(value: String) : this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  val shiftMean = new BooleanParam(
    this, "shiftMean","Whenever to substract mean")

  def isShiftMean : Boolean = $(shiftMean)
  def setShiftMean(value: Boolean) : this.type = set(shiftMean, value)

  setDefault(shiftMean -> true)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("LinearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {

    // Used to convert untyped dataframes to datasets with vectors
    implicit val encoder : Encoder[Vector] = ExpressionEncoder()

    val listCols = List($(inputCol), "label")
    //val listCols = List($(inputCol))
    //val vectors: Dataset[Vector] = dataset.select(listCols.map(m=>col(m)):_*).as[Vector]

    val allListCols = List($(inputCol), "constCol", "label")
    val vectors2: Dataset[Row] = dataset.select("*").withColumn("constCol", lit(1.0))

    val vectors4 = vectors2.select(allListCols.map(m=>col(m)):_*)
    val assembler = new VectorAssembler()
      .setInputCols(allListCols.toArray)
      .setOutputCol("all_features")

    val vectors3 = assembler.transform(vectors4)

    val vectors5 = vectors3.map(_.getAs[Vector]("all_features"))
    vectors5.foreach(v => println(v))
    //val vectors3 = vectors2.map(row => row)

    //val labels: Dataset[Vector] = dataset.select(dataset("label").as[Vector])

    val nFeat: Int = AttributeGroup.fromStructField((dataset.schema($(inputCol)))).numAttributes.getOrElse(
      vectors5.first().size
    ) - 1

    val weights0: Vector = Vectors.zeros(nFeat)

    val learnRate = 0.001

    val maxIter = 1000
    def descent(weights: Vector, iterCount: Int): Vector ={

      if (iterCount > 0){
        val whatisit = vectors5.rdd.map((x: Vector) =>
          2.0 * learnRate * (x.asBreeze(0 until nFeat).toDenseVector.dot(weights.asBreeze) - x.asBreeze(nFeat)) * x.asBreeze(0 until nFeat).toVector
        )

        //    val whatisit = vectors2.rdd.map((x: Vector) =>
        //      2.0 * learnRate * (x.asBreeze(0 to nFeat-2).toDenseVector.dot(weights.asBreeze) - x.asBreeze(nFeat-1))* x.asBreeze
        //    )

        val whatisit2 = whatisit.reduce((a, b) => a + b)

        descent(Vectors.fromBreeze(weights.asBreeze - whatisit2), iterCount - 1)
      } else {
        weights
      }
    }

    val weightsFound = descent(weights0, maxIter)

    copyValues(new LinearRegressionModel(
      Vectors.dense(1.0, 1.0),
      Vectors.dense(1.0))).setParent(this)

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
    val transformUdf = if (isShiftMean) {
      dataset.sqlContext.udf.register(uid + "_transform",
      (x : Vector) => {
        Vectors.fromBreeze((x.asBreeze - bslope) /:/ bintercept)
      })
    } else {
      dataset.sqlContext.udf.register(uid + "_transform",
        (x : Vector) => {
          Vectors.fromBreeze((x.asBreeze) /:/ bintercept)
        })
    }

    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
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
