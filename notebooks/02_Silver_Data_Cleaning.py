# Databricks notebook source
# MAGIC %md
# MAGIC Read Bronze Table

# COMMAND ----------

bronze_df = spark.table("bronze_traffic_data")

display(bronze_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Inspect the Schema

# COMMAND ----------

bronze_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Clean the Data

# COMMAND ----------

from pyspark.sql.functions import col, to_timestamp

silver_df = bronze_df.withColumn(
    "date_time",
    to_timestamp(col("date_time"))
)

# COMMAND ----------

silver_df = silver_df.dropna()

# COMMAND ----------

silver_df = silver_df.dropDuplicates()

# COMMAND ----------

# MAGIC %md
# MAGIC Feature Engineering

# COMMAND ----------

from pyspark.sql.functions import hour, dayofweek, month

silver_df = silver_df.withColumn("hour", hour("date_time")) \
                     .withColumn("day_of_week", dayofweek("date_time")) \
                     .withColumn("month", month("date_time"))

# COMMAND ----------

# MAGIC %md
# MAGIC Save Silver Table

# COMMAND ----------

silver_df.write.format("delta") \
.mode("overwrite") \
.saveAsTable("silver_traffic_data")

# COMMAND ----------

# MAGIC %md
# MAGIC Verify Silver Table

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM silver_traffic_data LIMIT 10