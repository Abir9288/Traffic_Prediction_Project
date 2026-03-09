# Databricks notebook source
# MAGIC %md
# MAGIC Import Libraries

# COMMAND ----------

from pyspark.sql.functions import col, when, lag, sin, cos
from pyspark.sql.window import Window
from math import pi
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor
from xgboost.spark import SparkXGBRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
from pyspark.sql.functions import coalesce, lit

# COMMAND ----------

df = spark.table("silver_traffic_data")

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Select Features for Prediction

# COMMAND ----------

# Temporal Features
df = df.withColumn("is_weekend", when(col("day_of_week").isin([6,7]),1).otherwise(0))
df = df.withColumn("rush_hour", when((col("hour").between(7,9)) | (col("hour").between(16,19)),1).otherwise(0))

# Weather-Derived Features
df = df.withColumn("precipitation", col("rain_1h")+col("snow_1h"))
df = df.withColumn("is_rain", when(col("rain_1h")>0,1).otherwise(0))
df = df.withColumn("is_snow", when(col("snow_1h")>0,1).otherwise(0))
df = df.withColumn("cloud_category",
                   when(col("clouds_all")<=30,"low")
                   .when(col("clouds_all")<=70,"medium")
                   .otherwise("high"))

# Lag Features
w = Window.orderBy("date_time")

df = df.withColumn("traffic_1h_ago", coalesce(lag("traffic_volume", 1).over(w), lit(0)))
df = df.withColumn("traffic_24h_ago", coalesce(lag("traffic_volume", 24).over(w), lit(0)))

# Interaction Features
df = df.withColumn("hour_rush_precip", col("rush_hour")*col("precipitation"))
df = df.withColumn("temp_precip", col("temp")*col("precipitation"))

# Cyclical Features
df = df.withColumn("hour_sin", sin(col("hour")*2*pi/24))
df = df.withColumn("hour_cos", cos(col("hour")*2*pi/24))
df = df.withColumn("day_sin", sin(col("day_of_week")*2*pi/7))
df = df.withColumn("day_cos", cos(col("day_of_week")*2*pi/7))

# COMMAND ----------

df.printSchema()  # confirm 'traffic_1h_ago' and 'traffic_24h_ago' are present

# COMMAND ----------

df.select("traffic_volume", "traffic_1h_ago", "traffic_24h_ago").show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC Handle Categorical Feature

# COMMAND ----------

# StringIndexer + OneHotEncoder
cloud_indexer = StringIndexer(inputCol="cloud_category", outputCol="cloud_category_index")
df = cloud_indexer.fit(df).transform(df)

cloud_encoder = OneHotEncoder(inputCols=["cloud_category_index"], outputCols=["cloud_category_vec"])
df = cloud_encoder.fit(df).transform(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Assemble Features for ML

# COMMAND ----------

feature_cols = [
    "temp", "rain_1h", "snow_1h", "clouds_all", "hour", "day_of_week", "month",
    "is_weekend", "rush_hour", "precipitation", "is_rain", "is_snow",
    "traffic_1h_ago", "traffic_24h_ago", "hour_rush_precip", "temp_precip",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "cloud_category_vec"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
df = assembler.transform(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Split Training & Testing Data

# COMMAND ----------

train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC Train Random Forest Model

# COMMAND ----------

rf = RandomForestRegressor(featuresCol="features", labelCol="traffic_volume", numTrees=100, maxDepth=10)
rf_model = rf.fit(train_data)

# Predictions
rf_predictions = rf_model.transform(test_data)

# Evaluate
evaluator = RegressionEvaluator(labelCol="traffic_volume", predictionCol="prediction", metricName="rmse")
rf_rmse = evaluator.evaluate(rf_predictions)
print("Random Forest RMSE:", rf_rmse)

# COMMAND ----------

# MAGIC %md
# MAGIC Train XGBoost Model

# COMMAND ----------

import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error

# Convert a sample to Pandas (avoid memory issues)
train_pd = train_data.select("features", "traffic_volume").toPandas()
X_train = np.vstack(train_pd['features'].values)
y_train = train_pd['traffic_volume'].values

test_pd = test_data.select("features", "traffic_volume").toPandas()
X_test = np.vstack(test_pd['features'].values)
y_test = test_pd['traffic_volume'].values

xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
xgb_rmse = np.sqrt(mse) 
print("RMSE:", xgb_rmse)

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Visualization — Compare RF vs XGBoost**

# COMMAND ----------

import pandas as pd

# Convert to Pandas for plotting
rf_pred_df  = rf_predictions.select("traffic_volume","prediction").toPandas()
xgb_pred_df = pd.DataFrame({
    "traffic_volume": y_test,
    "prediction": y_pred
})

# RMSE comparison
models = ["Random Forest","XGBoost"]
rmse_values = [rf_rmse, xgb_rmse]

# Combined figure
fig, axes = plt.subplots(1,3, figsize=(20,6))

# RF scatter
axes[0].scatter(rf_pred_df["traffic_volume"], rf_pred_df["prediction"], alpha=0.5, color='blue')
axes[0].plot([0,7000],[0,7000], color='red', linestyle='--')
axes[0].set_xlabel("Actual Traffic")
axes[0].set_ylabel("Predicted Traffic")
axes[0].set_title("Random Forest")

# XGBoost scatter
axes[1].scatter(xgb_pred_df["traffic_volume"], xgb_pred_df["prediction"], alpha=0.5, color='green')
axes[1].plot([0,7000],[0,7000], color='red', linestyle='--')
axes[1].set_xlabel("Actual Traffic")
axes[1].set_ylabel("Predicted Traffic")
axes[1].set_title("XGBoost")

# RMSE bar chart
bars = axes[2].bar(models, rmse_values, color=["blue","green"])
axes[2].set_title("RMSE Comparison")
axes[2].set_ylabel("RMSE")
axes[2].set_ylim(0, max(rmse_values)*1.1)

for bar, rmse in zip(bars, rmse_values):
    yval = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2.0, yval + 20, f"{rmse:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **MLflow Tracking & Logging**

# COMMAND ----------

import mlflow
import mlflow.spark

# UC volume path
uc_volume_path = "/Volumes/workspace/default/mlflow_uc"

with mlflow.start_run(run_name="Traffic_RF"):
    # Log parameters
    mlflow.log_param("numTrees", 100)
    mlflow.log_param("maxDepth", 10)
    
    # Log metrics
    mlflow.log_metric("rmse", rf_rmse)
    
    # Log Spark ML model
    mlflow.spark.log_model(
        rf_model,
        artifact_path="rf_model",
        dfs_tmpdir=uc_volume_path
    )
    
    print("Random Forest run logged successfully")

# COMMAND ----------

import mlflow
import pickle

uc_volume_path = "/Volumes/workspace/default/mlflow_uc"

# Save model locally
xgb_model_path = "/tmp/xgb_model.pkl"
with open(xgb_model_path, "wb") as f:
    pickle.dump(xgb_model, f)

# Log XGBoost model as an artifact
with mlflow.start_run(run_name="Traffic_XGBoost"):
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_metric("rmse", xgb_rmse)
    
    mlflow.log_artifact(xgb_model_path, artifact_path="xgb_model_pickle")
    
    print("XGBoost run logged successfully as pickle ")

# COMMAND ----------

# Random Forest predictions (Spark DF, works as is)
rf_predictions.select(
    "prediction", "features", "traffic_volume"
).write.format("delta").mode("overwrite").saveAsTable("gold_traffic_predictions_rf")

# XGBoost predictions (convert Pandas to Spark)
xgb_pred_spark = spark.createDataFrame(xgb_pred_df)

xgb_pred_spark.write.format("delta").mode("overwrite").saveAsTable("gold_traffic_predictions_xgb")