# Databricks notebook source
from pyspark.sql import functions as F

df = spark.table("workspace.default.flight_silver")

print("Rows:", df.count())
print("Columns:", len(df.columns))

# COMMAND ----------

display(
    df.groupBy("flight_status")
      .agg(
          F.count("*").alias("flights")
      )
      .withColumn(
          "percentage",
          F.round(
              F.col("flights") / df.count() * 100,
              2
          )
      )
      .orderBy(F.desc("flights"))
)

# COMMAND ----------

completed = df.filter(
    F.col("flight_status") == "Completed"
)

display(
    completed.groupBy("is_delayed")
             .count()
             .withColumn(
                 "percentage",
                 F.round(
                     F.col("count") / completed.count() * 100,
                     2
                 )
             )
)

# COMMAND ----------

display(
    completed.select(
        "dep_delay",
        "arr_delay",
        "taxi_out",
        "taxi_in",
        "air_time",
        "distance"
    ).summary()
)

# COMMAND ----------

airline_stats = (
    completed
    .groupBy("op_unique_carrier")
    .agg(
        F.count("*").alias("flights"),
        F.round(F.avg("arr_delay"), 2).alias("avg_arr_delay"),
        F.round(F.expr("percentile(arr_delay, 0.5)"), 2).alias("median_arr_delay"),
        F.round(F.avg("is_delayed") * 100, 2).alias("delay_rate_pct")
    )
    .orderBy(F.desc("delay_rate_pct"))
)

display(airline_stats)

# COMMAND ----------

airport_stats = (
    completed
    .groupBy(
        "origin",
        "origin_city_name",
        "origin_state_nm"
    )
    .agg(
        F.count("*").alias("flights"),
        F.round(F.avg("dep_delay"), 2).alias("avg_dep_delay"),
        F.round(F.avg("is_delayed") * 100, 2).alias("delay_rate_pct")
    )
    .filter(F.col("flights") >= 10000)
    .orderBy(F.desc("delay_rate_pct"))
)

display(airport_stats.limit(20))

# COMMAND ----------

display(
    completed
    .groupBy("departure_period")
    .agg(
        F.count("*").alias("flights"),
        F.round(F.avg("arr_delay"), 2).alias("avg_arr_delay"),
        F.round(F.avg("is_delayed") * 100, 2).alias("delay_rate_pct")
    )
    .orderBy("delay_rate_pct")
)

# COMMAND ----------

display(
    completed
    .groupBy("scheduled_dep_hour")
    .agg(
        F.count("*").alias("flights"),
        F.round(F.avg("arr_delay"), 2).alias("avg_arr_delay"),
        F.round(F.avg("is_delayed") * 100, 2).alias("delay_rate_pct")
    )
    .orderBy("scheduled_dep_hour")
)

# COMMAND ----------

route_stats = (
    completed
    .groupBy("route")
    .agg(
        F.count("*").alias("flights"),
        F.round(F.avg("arr_delay"), 2).alias("avg_arr_delay"),
        F.round(F.avg("is_delayed") * 100, 2).alias("delay_rate_pct")
    )
    .filter(F.col("flights") >= 1000)
    .orderBy(F.desc("delay_rate_pct"))
)

display(route_stats.limit(20))

# COMMAND ----------

delay_causes = completed.filter(
    F.col("arr_delay") >= 15
)

display(
    delay_causes.select(
        F.sum("carrier_delay").alias("carrier_delay"),
        F.sum("weather_delay").alias("weather_delay"),
        F.sum("nas_delay").alias("nas_delay"),
        F.sum("security_delay").alias("security_delay"),
        F.sum("late_aircraft_delay").alias("late_aircraft_delay")
    )
)

# COMMAND ----------

display(
    completed
    .groupBy("month")
    .agg(
        F.count("*").alias("flights"),
        F.round(F.avg("arr_delay"), 2).alias("avg_arr_delay"),
        F.round(F.avg("is_delayed") * 100, 2).alias("delay_rate_pct")
    )
    .orderBy("month")
)

# COMMAND ----------

distance_analysis = (
    completed
    .withColumn(
        "distance_bucket",
        F.when(F.col("distance") < 500, "<500")
         .when(F.col("distance") < 1000, "500-999")
         .when(F.col("distance") < 1500, "1000-1499")
         .when(F.col("distance") < 2500, "1500-2499")
         .otherwise("2500+")
    )
    .groupBy("distance_bucket")
    .agg(
        F.count("*").alias("flights"),
        F.round(F.avg("arr_delay"), 2).alias("avg_arr_delay"),
        F.round(F.avg("is_delayed") * 100, 2).alias("delay_rate_pct")
    )
)

display(distance_analysis)

# COMMAND ----------

