# Databricks notebook source
df = spark.table("workspace.default.flight_data_2024")

print("Rows:", df.count())
print("Columns:", len(df.columns))

df.printSchema()

# COMMAND ----------

display(df.limit(10))

# COMMAND ----------

from pyspark.sql import functions as F

missing = df.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in df.columns
])

display(missing)

# COMMAND ----------

duplicate_count = (
    df.groupBy(df.columns)
      .count()
      .filter(F.col("count") > 1)
      .count()
)

print("Duplicate rows:", duplicate_count)

# COMMAND ----------

display(
    df.select(
        "dep_delay",
        "arr_delay",
        "cancelled",
        "diverted"
    ).summary()
)

# COMMAND ----------

display(
    df.groupBy("cancelled")
      .count()
      .orderBy("cancelled")
)

# COMMAND ----------

display(
    df.groupBy("diverted")
      .count()
      .orderBy("diverted")
)

# COMMAND ----------

display(
    df.groupBy("op_unique_carrier")
      .count()
      .orderBy(F.desc("count"))
)

# COMMAND ----------

display(
    df.groupBy("origin")
      .count()
      .orderBy(F.desc("count"))
      .limit(20)
)

# COMMAND ----------

df = spark.table("workspace.default.flight_data_2024")

(
    df.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("workspace.default.flight_bronze")
)

# COMMAND ----------

bronze_df = spark.table("workspace.default.flight_bronze")

print("Rows:", bronze_df.count())
print("Columns:", len(bronze_df.columns))

# COMMAND ----------

display(
    spark.sql("""
        DESCRIBE DETAIL workspace.default.flight_bronze
    """)
)

# COMMAND ----------

display(
    bronze_df
    .groupBy("cancelled", "diverted")
    .agg(
        F.count("*").alias("flights"),
        F.sum(F.col("arr_delay").isNull().cast("int")).alias("missing_arr_delay"),
        F.sum(F.col("dep_delay").isNull().cast("int")).alias("missing_dep_delay"),
        F.sum(F.col("actual_elapsed_time").isNull().cast("int")).alias("missing_actual_elapsed"),
        F.sum(F.col("air_time").isNull().cast("int")).alias("missing_air_time")
    )
    .orderBy("cancelled", "diverted")
)

# COMMAND ----------

display(
    bronze_df
    .groupBy("cancellation_code")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

display(
    bronze_df
    .filter(F.col("cancelled") == 1)
    .select(
        "fl_date",
        "op_unique_carrier",
        "origin",
        "dest",
        "cancellation_code",
        "diverted"
    )
    .limit(20)
)

# COMMAND ----------

