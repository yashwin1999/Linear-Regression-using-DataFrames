# Spark Closed-Form & Outer-Product Linear Regression (Scala)

**Suggested project name:** `Spark Closed-Form & Outer-Product Linear Regression (Scala)`  
**Suggested repo name:** `spark-closedform-linear-regression`

---

## Project Summary

This project implements and compares two ways to compute linear regression parameters (θ) using Apache Spark and Breeze in Scala:

1. **Closed-form (Normal equations)** — compute \(\theta = (X^T X)^{+} X^T y\) where \(^{+}\) is the pseudoinverse (computed using Breeze `pinv`).  
2. **Outer-product based** — compute \(X^T X\) by summing outer products of row-vectors (\(\sum v_i v_i^T\)) and then solve for θ with the pseudoinverse.

The code is written as a Databricks/Spark notebook (Scala) and uses Spark DataFrames/RDDs to compute intermediate aggregated results, then collects the small dense matrices/vectors to the driver and uses Breeze for dense linear algebra operations.

A small test dataset is included in the notebook and the housing dataset (`/FileStore/tables/housing.csv`) is used to demonstrate the methods on real data.

---

## Project Structure (recommended)

```
spark-closedform-linear-regression/
├── README.md                         # This file
├── ClosedFormLinearRegression.scala  # Scala source / exported notebook
├── data/
│   └── housing.csv                   # (optional) dataset, or upload to DBFS/FileStore
├── .gitignore
└── .gitattributes
```

> **Note:** If you use Databricks, keep the notebook as a Databricks notebook and export the runnable `.scala` or `.dbc` file for the repository.

---

## Requirements

- Apache Spark (2.4+ or 3.x)
- Scala (2.11 / 2.12 matching your Spark build)
- Breeze linear algebra library (e.g. `org.scalanlp:breeze_2.12:1.2` or compatible)
- Databricks Runtime (optional — notebook runs there) or `spark-shell` / `sbt` for local runs

---

## How the Code Works (analysis)

### Data ingestion & representation
- The notebook reads input rows as `Array[Double]` where the **last element is the target `y`** and the preceding elements are features.  
- `createCOOMatrix` converts the dataset into a Coordinate (COO) style DataFrame: `(rowIndex, colIndex, value)`. It also **adds a bias column** (value 1.0 at the last column index). The function also returns a DataFrame `y_rdd` with `(rowIndex, value)` for targets.
- For the `housing.csv` dataset, the notebook expects data rows parsed into arrays of doubles and filtered to the correct length; it reads `"/FileStore/tables/housing.csv"` (Databricks FileStore path).

### Closed-form method
- Uses SQL-style self-join on the COO DataFrame to compute `X^T X`:
  ```sql
  SELECT a.colIndex AS i, b.colIndex AS j, SUM(a.value * b.value) AS value
  FROM xDF a JOIN xDF b ON a.rowIndex = b.rowIndex
  GROUP BY i, j ORDER BY i, j
  ```
- Collects the aggregated `X^T X` entries to the driver and fills a Breeze `DenseMatrix`.
- Computes pseudoinverse `pinv(X^T X)` and then `theta = pinv(X^T X) * (X^T y)` using Breeze on the driver.

### Outer-product method
- Groups rows by `rowIndex` and builds a dense vector `v` of that row's features (including bias), then computes `v * v.t` and sums across rows to build `X^T X` incrementally:
  ```scala
  // per row: v = [x1, x2, ..., 1], then outerSum += v * v.t
  ```
- Similarly collects `X^T y` via a SQL-style aggregation, builds a Breeze vector on driver, then computes `theta = pinv(outerSum) * (X^T y)`.

### Small test example
The notebook includes a small test dataset:
```text
[1.0, 2.0, 9.0]
[2.0, 3.0, 13.0]
[3.0, 4.0, 17.0]
```
For this small dataset (two features + bias), the closed-form and outer-product methods produce identical θ:

```
θ = [1.0, 3.0, 2.0]  # coefficients for feature1, feature2, bias (last entry)
```
This means the learned model fits the test samples exactly: `y = 1*x1 + 3*x2 + 2`.

---

## Strengths & Limitations (analysis)

### Strengths
- **Correctness**: Both methods compute the same mathematical result (within numerical tolerance).  
- **Clear comparison**: Shows two distinct ways of constructing `X^T X` from distributed data and how to bring it into Breeze for solving.  
- **Databricks-friendly**: Uses DataFrames and small-driver Breeze operations — easy to run on a notebook for educational/demo purposes.

### Limitations / Scalability Concerns
- **Driver-side collection**: The design collects the full `d x d` matrix (`X^T X`) and `d` vector (`X^T y`) to the driver. This is fine when feature dimension `d` is small (e.g., tens or a few hundreds) but **will not scale** if `d` grows to thousands — memory and computation on the driver become the bottleneck.
- **Matrix inversion stability**: Using `pinv` helps with singular matrices, but numerical stability and performance might be problematic for ill-conditioned matrices. Consider **regularization** (Ridge — add λI) to stabilize the inversion.
- **Performance tradeoffs**: The outer-product method reduces some shuffle compared to a full join approach but still requires per-row aggregation and collecting results. For very large datasets, use distributed linear algebra primitives (Spark MLlib `RowMatrix` / `BlockMatrix`) or iterative methods (like gradient descent or LBFGS) that better fit the distributed environment.
- **Dependence on Breeze**: Requires Breeze library on the driver. In Databricks, you can attach Breeze as a library; for `spark-submit`/`sbt` include the dependency.

---

## Suggested Improvements / Next Steps
1. **Add regularization**: compute `(X^T X + λI)` before inversion to stabilize solution.  
2. **Use distributed linear algebra**: If `d` becomes large, switch to MLlib's `RowMatrix` / `BlockMatrix` or use iterative solvers to avoid collecting dense matrices.  
3. **Broadcast small matrices**: If you can compute parts of `X^T X` on executors and broadcast them when solving, you can reduce driver memory pressure.  
4. **Profile and benchmark**: Measure wall time for each approach with varying `d` and `n` and show tradeoffs. The notebook already measures time for the Breeze solve step; broaden these measurements.  
5. **Add unit tests**: Add small deterministic tests for the helper functions.  
6. **Add data preprocessing**: Normalization / feature scaling will help numerical stability.

---

## How to Run (Databricks Notebook)

1. Create a new Scala notebook in Databricks.  
2. If using `housing.csv`, upload it to **DBFS/FileStore** (`/FileStore/tables/housing.csv`) or update the code to point to your data path.  
3. Ensure Breeze is available: in Databricks add a Maven library or upload the JAR. Example Maven coordinates (match your Scala version):
   - `org.scalanlp:breeze_2.12:1.2`  
   (If runtime mismatch occurs, pick a Breeze artifact matching your Scala version.)  
4. Paste the notebook code into the notebook cells and run cells in order. The small test will print `θ` and timing metrics for both methods. For the housing dataset, final θ and timing will be printed too.

---

## How to Run Locally (`spark-shell` or `sbt`)

### Option A: `spark-shell` (quick experiment)
1. Download the Breeze JAR that matches your Scala version and put it on the classpath, or use `--packages` if available:
```bash
spark-shell --packages org.scalanlp:breeze_2.12:1.2
```
2. In the shell, paste relevant function definitions from the notebook or `:load ClosedFormLinearRegression.scala` if saved to a file. Ensure the data path exists locally or change to a local CSV path and parsing logic.

### Option B: Build with `sbt` and run with `spark-submit`
1. Create `build.sbt` with dependencies:
```sbt
name := "SparkClosedFormRegression"
version := "0.1"
scalaVersion := "2.12.15"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % "3.3.0",       // match your Spark version
  "org.scalanlp" %% "breeze" % "1.2"
)
```
2. Package the application and run with `spark-submit` (ensure Spark and Scala versions match):
```bash
sbt package
spark-submit --class your.main.Class --master local[*] target/scala-2.12/sparkclosedformregression_2.12-0.1.jar
```

---

## Quick Start: Saving & Uploading to GitHub (Windows CMD)

From your project folder (where `ClosedFormLinearRegression.scala` and `README.md` live):

```cmd
cd "C:\Path\To\spark-closedform-linear-regression"

rem initialize git (if not already)
git init
git branch -M main

rem create useful metadata files
echo > .gitignore
notepad .gitignore
rem paste recommended contents, save & close

git add .
git commit -m "Initial commit: closed-form vs outer-product linear regression notebook + README"

rem create a GitHub repo (via web UI or gh cli), then add origin:
git remote add origin https://github.com/<your-username>/spark-closedform-linear-regression.git

git push -u origin main
```

**Recommended `.gitignore` contents** (Scala/Spark/IDE-friendly):
```
target/
*.class
*.log
project/
.metals/
.bloop/
.idea/
.vscode/
.DS_Store
```
**Recommended `.gitattributes`** (to enforce LF):
```
* text=auto eol=lf
```

---

## Example Output (small test)

From the included 3-row test dataset, the notebook should print something like:

```
[Test] θ (Main method): DenseVector(1.0, 3.0, 2.0)
[Test] θ (Outer method): DenseVector(1.0, 3.0, 2.0)
[Test] Are θ₁ and θ₂ approximately equal? true
```

---

## Author & License

Created by Yashwin Bangalore Subramani.  
License: choose one (e.g., MIT) — add `LICENSE` file to repository if you want to open-source it.

---

## Final notes / Reminders

- If you plan to run experiments on large data, avoid collecting large aggregated matrices to the driver; instead use distributed math tools or iterative solvers.  
- Add comments in the Scala notebook so readers can quickly follow each step.  
- If you want, I can also:
  - Convert the notebook into a `.scala` source file ready for `spark-shell` or `sbt`.  
  - Create the `.gitignore` and `.gitattributes` files for you and make them downloadable.  
  - Prepare a short `README` tailored for Databricks specifically (with import/export steps).
