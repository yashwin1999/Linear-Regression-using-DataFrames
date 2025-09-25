// Databricks notebook source
// MAGIC %md
// MAGIC ## closed-form

// COMMAND ----------

import org.apache.spark.sql.functions._
import breeze.linalg._
import org.apache.spark.sql.{Dataset, DataFrame}

spark.sql("set spark.sql.legacy.allowUntypedScalaUDF=true")

// COMMAND ----------

def createCOOMatrix(data: Dataset[Array[Double]]) = {
  // Dynamically slice based on number of features in the input data
  val numFeatures = data.head().length - 1  // assuming last value is target
  val X = data.map(row => row.slice(0, numFeatures))  // Use only features
  val y = data.map(row => row.last)  // The target is the last element

  val X_rdd = X.rdd.zipWithIndex()
  val xCOO = X_rdd.flatMap { case (features, rowIndex) =>
    val entries = features.zipWithIndex.map { case (value, colIndex) =>
      (rowIndex, colIndex.toLong, value)
    }
    val bias = Seq((rowIndex, features.length.toLong, 1.0))  // Add bias term
    entries ++ bias
  }.toDF("rowIndex", "colIndex", "value")

  val y_rdd = y.rdd.zipWithIndex().map { case (value, rowIndex) =>
    (rowIndex, value)
  }.toDF("rowIndex", "value")

  (xCOO, y_rdd)
}


// Closed-Form Function
def computeThetaClosedForm(xCOO: DataFrame, y_rdd: DataFrame): (DenseVector[Double], Double) = {
  xCOO.createOrReplaceTempView("xDF")
  y_rdd.createOrReplaceTempView("yDF")

  val xT_x = spark.sql("""
    SELECT a.colIndex AS i, b.colIndex AS j,
           SUM(a.value * b.value) AS value
    FROM xDF a JOIN xDF b ON a.rowIndex = b.rowIndex
    GROUP BY i, j ORDER BY i, j
  """)

  val numFeatures = xCOO.selectExpr("colIndex").distinct().count().toInt
  val xT_x_breeze = DenseMatrix.zeros[Double](numFeatures, numFeatures)
  xT_x.collect().foreach { row =>
    val i = row.getAs[Long]("i").toInt
    val j = row.getAs[Long]("j").toInt
    val value = row.getAs[Double]("value")
    xT_x_breeze(i, j) = value
  }

  val xT_x_inv = pinv(xT_x_breeze)

  val xT_y = spark.sql("""
    SELECT x.colIndex AS colIndex, SUM(x.value * y.value) AS value
    FROM xDF x JOIN yDF y ON x.rowIndex = y.rowIndex
    GROUP BY colIndex ORDER BY colIndex
  """)

  val xT_y_breeze = DenseVector.zeros[Double](numFeatures)
  xT_y.collect().foreach { row =>
    val idx = row.getAs[Long]("colIndex").toInt
    val value = row.getAs[Double]("value")
    xT_y_breeze(idx) = value
  }

  val t1 = System.nanoTime()
  val theta = xT_x_inv * xT_y_breeze
  val t2 = System.nanoTime()
  val time = (t2 - t1) / 1e9

  (theta, time)
}

// Outer Product Function
def computeThetaOuterProduct(xCOO: DataFrame, xT_y_breeze: DenseVector[Double]): (DenseVector[Double], Double) = {
  xCOO.createOrReplaceTempView("xDF")

  val numFeatures = xCOO.selectExpr("colIndex").distinct().count().toInt

  val rowGrouped = xCOO.groupBy("rowIndex")
    .agg(collect_list(struct("colIndex", "value")).as("features"))
    .collect()

  val xOuterSum = DenseMatrix.zeros[Double](numFeatures, numFeatures)
  rowGrouped.foreach { row =>
    val features = row.getAs[Seq[Row]]("features")
    val v = DenseVector.zeros[Double](numFeatures)
    features.foreach { r =>
      val colIdx = r.getAs[Long]("colIndex").toInt
      val value = r.getAs[Double]("value")
      v(colIdx) = value
    }
    xOuterSum += v * v.t
  }

  val xOuterInv = pinv(xOuterSum)

  val t3 = System.nanoTime()
  val theta2 = xOuterInv * xT_y_breeze
  val t4 = System.nanoTime()
  val time = (t4 - t3) / 1e9

  (theta2, time)
}


// COMMAND ----------

// Test block using a small custom dataset
import org.apache.spark.sql.expressions.Window
import spark.implicits._

// Small dataset: 3 rows, 2 features + 1 target
val testData = Seq(
  Array(1.0, 2.0, 9.0),
  Array(2.0, 3.0, 13.0),
  Array(3.0, 4.0, 17.0)
).toDS()

val (xCOO_test, y_rdd_test) = createCOOMatrix(testData)

val (theta_test_main, time_test_main) = computeThetaClosedForm(xCOO_test, y_rdd_test)
println(f"[Test] θ (Main method): $theta_test_main")
println(f"[Test] Time (Main method): $time_test_main%.4f seconds")

// Convert xCOO_test and y_rdd_test to DataFrames for outer product method
val xDF_test = xCOO_test.toDF("rowIndex", "colIndex", "value")
val yDF_test = y_rdd_test.withColumn("rowIndex", row_number().over(Window.orderBy(lit(1))) - 1).select("rowIndex", "value")
xDF_test.createOrReplaceTempView("xDF_test")
yDF_test.createOrReplaceTempView("yDF_test")

val xT_y_test = spark.sql("""
  SELECT x.colIndex AS colIndex, SUM(x.value * y.value) AS value
  FROM xDF_test x JOIN yDF_test y ON x.rowIndex = y.rowIndex
  GROUP BY colIndex ORDER BY colIndex
""")

val numFeaturesTest = xT_y_test.count().toInt
val xT_y_breeze_test = DenseVector.zeros[Double](numFeaturesTest)
xT_y_test.collect().foreach { row =>
  xT_y_breeze_test(row.getLong(0).toInt) = row.getDouble(1)
}


val (theta_test_outer, time_test_outer) = computeThetaOuterProduct(xCOO_test, xT_y_breeze_test)
println(f"[Test] θ (Outer method): $theta_test_outer")
println(f"[Test] Time (Outer method): $time_test_outer%.4f seconds")

val diff_test = norm(theta_test_main - theta_test_outer)
val equal_test = diff_test < 1e-6

println(f"[Test] Are θ₁ and θ₂ approximately equal? $equal_test")
println(f"[Test] Difference: $diff_test%.8f\n")


// COMMAND ----------

val raw = spark.read.textFile("/FileStore/tables/housing.csv")

val data = raw
  .filter("trim(value) != ''")
  .map(row => row.trim.split("\\s+").map(_.toDouble))
  .filter(_.length == 14)
  .cache()

val (xCOO, y_rdd) = createCOOMatrix(data)


// COMMAND ----------

val (theta1, timeMain) = computeThetaClosedForm(xCOO, y_rdd)
println(f"θ (Main method): $theta1")
println(f"Time (Main method): $timeMain%.4f seconds")


// COMMAND ----------

// Recompute X^T y to get xT_y_breeze for outer product method
val xT_y = spark.sql("""
  SELECT x.colIndex AS colIndex, SUM(x.value * y.value) AS value
  FROM xDF x JOIN yDF y ON x.rowIndex = y.rowIndex
  GROUP BY colIndex ORDER BY colIndex
""")

val numFeatures = xCOO.selectExpr("colIndex").distinct().count().toInt
val xT_y_breeze = DenseVector.zeros[Double](numFeatures)
xT_y.collect().foreach { row =>
  val idx = row.getAs[Long]("colIndex").toInt
  val value = row.getAs[Double]("value")
  xT_y_breeze(idx) = value
}


// COMMAND ----------

val (theta2, timeOuter) = computeThetaOuterProduct(xCOO, xT_y_breeze)
println(f"θ (Outer method): $theta2")
println(f"Time (Outer method): $timeOuter%.4f seconds")


// COMMAND ----------

val difference = norm(theta1 - theta2)
val equal = difference < 1e-4

println(f"Are θ₁ and θ₂ approximately equal? $equal")
println(f"Difference: $difference%.6f")

if (timeMain < timeOuter) {
  println("Closed-form method is faster.")
} else {
  println("Outer product method is faster.")
}


// COMMAND ----------

