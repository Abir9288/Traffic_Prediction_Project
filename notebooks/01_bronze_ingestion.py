# Databricks notebook source
# MAGIC %md
# MAGIC Read Dataset

# COMMAND ----------

df = spark.read.csv(
"/Volumes/workspace/default/traffic_data_volume/Metro_Interstate_Traffic_Volume.csv",
header=True,
inferSchema=True
)

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Create Bronze Delta Table

# COMMAND ----------

df.write.format("delta") \
.mode("overwrite") \
.saveAsTable("bronze_traffic_data")

# COMMAND ----------

# MAGIC %md
# MAGIC Verify Bronze Table

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM bronze_traffic_data LIMIT 10