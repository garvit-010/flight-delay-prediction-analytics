# Databricks notebook source
from pyspark.sql import functions as F

df = spark.table("workspace.default.flight_bronze")

# COMMAND ----------

df = df.withColumn(
    "flight_status",
    F.when(F.col("cancelled") == 1, "Cancelled")
     .when(F.col("diverted") == 1, "Diverted")
     .otherwise("Completed")
)

# COMMAND ----------

display(
    df.groupBy("flight_status")
      .count()
      .orderBy(F.desc("count"))
)

# COMMAND ----------

df = (
    df
    .withColumn(
        "scheduled_dep_hour",
        (F.col("crs_dep_time") / 100).cast("int")
    )
    .withColumn(
        "scheduled_dep_minute",
        (F.col("crs_dep_time") % 100).cast("int")
    )
)

# COMMAND ----------

display(
    df.select(
        "crs_dep_time",
        "scheduled_dep_hour",
        "scheduled_dep_minute"
    ).limit(20)
)

# COMMAND ----------

df = df.withColumn(
    "is_delayed",
    F.when(F.col("arr_delay") >= 15, 1)
     .when(F.col("arr_delay").isNotNull(), 0)
     .otherwise(None)
)

# COMMAND ----------

(
    df.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("workspace.default.flight_silver")
)

# COMMAND ----------

silver_df = spark.table("workspace.default.flight_silver")

print("Rows:", silver_df.count())
print("Columns:", len(silver_df.columns))

# COMMAND ----------

display(
    df.select(
        "fl_date",
        "op_unique_carrier",
        "origin",
        "dest",
        "crs_dep_time",
        "scheduled_dep_hour",
        "scheduled_dep_minute",
        "flight_status",
        "arr_delay",
        "is_delayed"
    ).limit(20)
)

# COMMAND ----------

display(
    df.filter(
        (F.col("crs_dep_time") < 0) |
        (F.col("crs_dep_time") > 2359) |
        (F.col("scheduled_dep_minute") > 59)
    ).select(
        "crs_dep_time",
        "scheduled_dep_hour",
        "scheduled_dep_minute"
    ).limit(20)
)

# COMMAND ----------

display(
    df.filter(F.col("distance") <= 0)
      .select(
          "origin",
          "dest",
          "distance"
      )
      .limit(20)
)

# COMMAND ----------

display(
    df.filter(
        ~F.col("cancelled").isin(0, 1) |
        ~F.col("diverted").isin(0, 1)
    ).select(
        "cancelled",
        "diverted"
    ).limit(20)
)

# COMMAND ----------

display(
    df.filter(
        (F.col("cancelled") == 1) &
        (F.col("diverted") == 1)
    ).select(
        "fl_date",
        "op_unique_carrier",
        "origin",
        "dest",
        "cancelled",
        "diverted"
    ).limit(20)
)

# COMMAND ----------

df = (
    df
    .withColumn(
        "crs_dep_time_clean",
        F.when(F.col("crs_dep_time") == 2400, 0)
         .otherwise(F.col("crs_dep_time"))
    )
    .withColumn(
        "scheduled_dep_hour",
        (F.col("crs_dep_time_clean") / 100).cast("int")
    )
    .withColumn(
        "scheduled_dep_minute",
        (F.col("crs_dep_time_clean") % 100).cast("int")
    )
)

# COMMAND ----------

display(
    df.filter(F.col("crs_dep_time") == 2400)
      .select(
          "crs_dep_time",
          "crs_dep_time_clean",
          "scheduled_dep_hour",
          "scheduled_dep_minute"
      )
)

# COMMAND ----------

display(
    df.filter(
        (F.col("crs_dep_time_clean") < 0) |
        (F.col("crs_dep_time_clean") > 2359) |
        (F.col("scheduled_dep_minute") > 59)
    )
)

# COMMAND ----------

df = df.withColumn(
    "route",
    F.concat_ws("_", F.col("origin"), F.col("dest"))
)

# COMMAND ----------

display(
    df.select(
        "origin",
        "dest",
        "route"
    ).limit(20)
)

# COMMAND ----------

df = df.withColumn(
    "departure_period",
    F.when(F.col("scheduled_dep_hour") < 6, "Night")
     .when(F.col("scheduled_dep_hour") < 12, "Morning")
     .when(F.col("scheduled_dep_hour") < 18, "Afternoon")
     .otherwise("Evening")
)

# COMMAND ----------

display(
    df.groupBy("departure_period")
      .count()
      .orderBy("departure_period")
)

# COMMAND ----------

display(
    df.filter(F.col("distance") <= 0)
      .select("origin", "dest", "distance")
      .limit(20)
)

# COMMAND ----------

display(
    df.filter(
        ~F.col("cancelled").isin(0, 1) |
        ~F.col("diverted").isin(0, 1)
    ).select(
        "cancelled",
        "diverted"
    ).limit(20)
)

# COMMAND ----------

display(
    df.filter(
        (F.col("cancelled") == 1) &
        (F.col("diverted") == 1)
    ).select(
        "fl_date",
        "op_unique_carrier",
        "origin",
        "dest",
        "cancelled",
        "diverted"
    ).limit(20)
)

# COMMAND ----------

display(
    df.select(
        "fl_date",
        "op_unique_carrier",
        "origin",
        "dest",
        "dep_delay",
        "arr_delay",
        "cancelled",
        "diverted"
    )
    .orderBy(F.desc("arr_delay"))
    .limit(20)
)

# COMMAND ----------

spark.sql("""
DROP TABLE IF EXISTS workspace.default.flight_silver
""")

# COMMAND ----------

print(
    spark.catalog.tableExists("workspace.default.flight_silver")
)

# COMMAND ----------

(
    df.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("workspace.default.flight_silver")
)

# COMMAND ----------

silver_df = spark.table("workspace.default.flight_silver")

print("Rows:", silver_df.count())
print("Columns:", len(silver_df.columns))

silver_df.printSchema()

# COMMAND ----------

silver_df = spark.table("workspace.default.flight_silver")

print("Rows:", silver_df.count())
print("Columns:", len(silver_df.columns))

# COMMAND ----------

