# Databricks notebook source
# MAGIC %sql
# MAGIC SELECT * FROM silver_traffic_data LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC Traffic by Hour (Peak Traffic)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold_traffic_by_hour AS
# MAGIC SELECT
# MAGIC     hour,
# MAGIC     AVG(traffic_volume) AS avg_traffic
# MAGIC FROM silver_traffic_data
# MAGIC GROUP BY hour
# MAGIC ORDER BY hour;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM gold_traffic_by_hour;

# COMMAND ----------

# MAGIC %md
# MAGIC Traffic by Day of Week

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold_traffic_by_day AS
# MAGIC SELECT
# MAGIC     day_of_week,
# MAGIC     AVG(traffic_volume) AS avg_traffic
# MAGIC FROM silver_traffic_data
# MAGIC GROUP BY day_of_week
# MAGIC ORDER BY day_of_week;

# COMMAND ----------

# MAGIC %md
# MAGIC Traffic by Weather Condition

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold_traffic_by_weather AS
# MAGIC SELECT
# MAGIC     weather_main,
# MAGIC     AVG(traffic_volume) AS avg_traffic
# MAGIC FROM silver_traffic_data
# MAGIC GROUP BY weather_main
# MAGIC ORDER BY avg_traffic DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC Create Visualizations

# COMMAND ----------

import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC Traffic by Hour

# COMMAND ----------

df_hour = spark.table("gold_traffic_by_hour").toPandas()

plt.figure()
plt.plot(df_hour["hour"], df_hour["avg_traffic"], marker='o')
plt.xlabel("Hour of Day")
plt.ylabel("Average Traffic Volume")
plt.title("Average Traffic by Hour")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Traffic by Day of Week

# COMMAND ----------

df_day = spark.table("gold_traffic_by_day").toPandas()

plt.figure()
plt.bar(df_day["day_of_week"], df_day["avg_traffic"])
plt.xlabel("Day of Week")
plt.ylabel("Average Traffic Volume")
plt.title("Traffic by Day of Week")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Traffic by Weather

# COMMAND ----------

df_weather = spark.table("gold_traffic_by_weather").toPandas()

plt.figure()
plt.bar(df_weather["weather_main"], df_weather["avg_traffic"])
plt.xlabel("Weather Condition")
plt.ylabel("Average Traffic Volume")
plt.title("Traffic by Weather Condition")
plt.xticks(rotation=45)
plt.show()