# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window

df = spark.table("workspace.default.flight_silver")

completed = df.filter(
    F.col("flight_status") == "Completed"
)

# COMMAND ----------

ml_df = completed

# COMMAND ----------

print("ML rows:", ml_df.count())

display(
    ml_df.groupBy("is_delayed")
         .count()
         .withColumn(
             "percentage",
             F.round(
                 F.col("count") / ml_df.count() * 100,
                 2
             )
         )
)

# COMMAND ----------

ml_df = ml_df.withColumn(
    "is_weekend",
    F.when(F.col("day_of_week").isin(1, 7), 1).otherwise(0)
)

# COMMAND ----------

ml_df = (
    ml_df
    .withColumn(
        "dep_hour_sin",
        F.sin(
            2 * F.pi() *
            F.col("scheduled_dep_hour") / 24
        )
    )
    .withColumn(
        "dep_hour_cos",
        F.cos(
            2 * F.pi() *
            F.col("scheduled_dep_hour") / 24
        )
    )
)

# COMMAND ----------

ml_df = (
    ml_df
    .withColumn(
        "month_sin",
        F.sin(
            2 * F.pi() *
            F.col("month") / 12
        )
    )
    .withColumn(
        "month_cos",
        F.cos(
            2 * F.pi() *
            F.col("month") / 12
        )
    )
)

# COMMAND ----------

route_daily = (
    ml_df
    .groupBy("fl_date", "route")
    .agg(
        F.count("*").alias("route_flights"),
        F.sum("is_delayed").alias("route_delayed")
    )
)

# COMMAND ----------

route_window = (
    Window
    .partitionBy("route")
    .orderBy("fl_date")
    .rowsBetween(
        Window.unboundedPreceding,
        -1
    )
)

# COMMAND ----------

route_daily = (
    route_daily
    .withColumn(
        "historical_route_flights",
        F.sum("route_flights").over(route_window)
    )
    .withColumn(
        "historical_route_delayed",
        F.sum("route_delayed").over(route_window)
    )
)

# COMMAND ----------

route_daily = route_daily.withColumn(
    "historical_route_delay_rate",
    F.when(
        F.col("historical_route_flights") > 0,
        F.col("historical_route_delayed") /
        F.col("historical_route_flights")
    )
)

# COMMAND ----------

ml_df = ml_df.join(
    route_daily.select(
        "fl_date",
        "route",
        "historical_route_flights",
        "historical_route_delay_rate"
    ),
    on=["fl_date", "route"],
    how="left"
)

# COMMAND ----------

origin_daily = (
    ml_df
    .groupBy("fl_date", "origin")
    .agg(
        F.count("*").alias("origin_flights"),
        F.sum("is_delayed").alias("origin_delayed")
    )
)

# COMMAND ----------

origin_window = (
    Window
    .partitionBy("origin")
    .orderBy("fl_date")
    .rowsBetween(
        Window.unboundedPreceding,
        -1
    )
)

# COMMAND ----------

origin_daily = (
    origin_daily
    .withColumn(
        "historical_origin_flights",
        F.sum("origin_flights").over(origin_window)
    )
    .withColumn(
        "historical_origin_delayed",
        F.sum("origin_delayed").over(origin_window)
    )
    .withColumn(
        "historical_origin_delay_rate",
        F.when(
            F.col("historical_origin_flights") > 0,
            F.col("historical_origin_delayed") /
            F.col("historical_origin_flights")
        )
    )
)

# COMMAND ----------

ml_df = ml_df.join(
    origin_daily.select(
        "fl_date",
        "origin",
        "historical_origin_flights",
        "historical_origin_delay_rate"
    ),
    on=["fl_date", "origin"],
    how="left"
)

# COMMAND ----------

carrier_daily = (
    ml_df
    .groupBy("fl_date", "op_unique_carrier")
    .agg(
        F.count("*").alias("carrier_flights"),
        F.sum("is_delayed").alias("carrier_delayed")
    )
)

# COMMAND ----------

carrier_window = (
    Window
    .partitionBy("op_unique_carrier")
    .orderBy("fl_date")
    .rowsBetween(
        Window.unboundedPreceding,
        -1
    )
)

# COMMAND ----------

carrier_daily = (
    carrier_daily
    .withColumn(
        "historical_carrier_flights",
        F.sum("carrier_flights").over(carrier_window)
    )
    .withColumn(
        "historical_carrier_delayed",
        F.sum("carrier_delayed").over(carrier_window)
    )
    .withColumn(
        "historical_carrier_delay_rate",
        F.when(
            F.col("historical_carrier_flights") > 0,
            F.col("historical_carrier_delayed") /
            F.col("historical_carrier_flights")
        )
    )
)

# COMMAND ----------

ml_df = ml_df.join(
    carrier_daily.select(
        "fl_date",
        "op_unique_carrier",
        "historical_carrier_flights",
        "historical_carrier_delay_rate"
    ),
    on=["fl_date", "op_unique_carrier"],
    how="left"
)

# COMMAND ----------

display(
    ml_df.select(
        "fl_date",
        "op_unique_carrier",
        "origin",
        "dest",
        "route",
        "scheduled_dep_hour",
        "month",
        "distance",
        "historical_route_delay_rate",
        "historical_origin_delay_rate",
        "historical_carrier_delay_rate",
        "is_delayed"
    ).orderBy("fl_date").limit(30)
)

# COMMAND ----------

ml_df = (
    ml_df
    .withColumn(
        "has_route_history",
        F.when(
            F.col("historical_route_flights").isNotNull() &
            (F.col("historical_route_flights") > 0),
            1
        ).otherwise(0)
    )
    .withColumn(
        "has_origin_history",
        F.when(
            F.col("historical_origin_flights").isNotNull() &
            (F.col("historical_origin_flights") > 0),
            1
        ).otherwise(0)
    )
    .withColumn(
        "has_carrier_history",
        F.when(
            F.col("historical_carrier_flights").isNotNull() &
            (F.col("historical_carrier_flights") > 0),
            1
        ).otherwise(0)
    )
)

# COMMAND ----------

display(
    ml_df
    .filter(F.col("fl_date") >= "2024-02-01")
    .select(
        "fl_date",
        "route",
        "historical_route_flights",
        "historical_route_delay_rate",
        "historical_origin_flights",
        "historical_origin_delay_rate",
        "historical_carrier_flights",
        "historical_carrier_delay_rate",
        "is_delayed"
    )
    .limit(30)
)

# COMMAND ----------

display(
    ml_df.select(
        F.min("historical_route_flights").alias("min_route_history"),
        F.max("historical_route_flights").alias("max_route_history"),
        F.min("historical_origin_flights").alias("min_origin_history"),
        F.max("historical_origin_flights").alias("max_origin_history"),
        F.min("historical_carrier_flights").alias("min_carrier_history"),
        F.max("historical_carrier_flights").alias("max_carrier_history")
    )
)

# COMMAND ----------

display(
    ml_df.groupBy("has_route_history")
         .count()
         .withColumn(
             "percentage",
             F.round(F.col("count") / ml_df.count() * 100, 2)
         )
)

# COMMAND ----------

ml_df = (
    ml_df
    .withColumn(
        "reliable_route_delay_rate",
        F.when(
            F.col("historical_route_flights") >= 20,
            F.col("historical_route_delay_rate")
        )
    )
    .withColumn(
        "reliable_origin_delay_rate",
        F.when(
            F.col("historical_origin_flights") >= 20,
            F.col("historical_origin_delay_rate")
        )
    )
    .withColumn(
        "reliable_carrier_delay_rate",
        F.when(
            F.col("historical_carrier_flights") >= 20,
            F.col("historical_carrier_delay_rate")
        )
    )
)

# COMMAND ----------

display(
    ml_df.select(
        "fl_date",
        "route",
        "historical_route_flights",
        "historical_route_delay_rate",
        "reliable_route_delay_rate",
        "origin",
        "historical_origin_flights",
        "reliable_origin_delay_rate",
        "op_unique_carrier",
        "historical_carrier_flights",
        "reliable_carrier_delay_rate",
        "is_delayed"
    )
    .orderBy("fl_date")
    .limit(50)
)

# COMMAND ----------

display(
    ml_df.select(
        F.count("*").alias("total"),
        F.sum(
            F.col("reliable_route_delay_rate").isNull().cast("int")
        ).alias("route_rate_unreliable"),
        F.sum(
            F.col("reliable_origin_delay_rate").isNull().cast("int")
        ).alias("origin_rate_unreliable"),
        F.sum(
            F.col("reliable_carrier_delay_rate").isNull().cast("int")
        ).alias("carrier_rate_unreliable")
    )
)

# COMMAND ----------

feature_columns = [
    # Temporal
    "fl_date",
    "month",
    "day_of_week",
    "scheduled_dep_hour",
    "scheduled_dep_minute",
    "departure_period",
    "is_weekend",
    "dep_hour_sin",
    "dep_hour_cos",
    "month_sin",
    "month_cos",

    # Flight / route
    "op_unique_carrier",
    "origin",
    "dest",
    "route",
    "distance",

    # Historical behavior
    "historical_route_flights",
    "reliable_route_delay_rate",
    "has_route_history",

    "historical_origin_flights",
    "reliable_origin_delay_rate",
    "has_origin_history",

    "historical_carrier_flights",
    "reliable_carrier_delay_rate",
    "has_carrier_history",

    # Target
    "is_delayed"
]

ml_features = ml_df.select(feature_columns)

# COMMAND ----------

(
    ml_features.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("workspace.default.flight_ml_features")
)

# COMMAND ----------

display(
    spark.table("workspace.default.flight_ml_features").limit(10)
)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

route_daily_7d = (
    ml_df
    .groupBy("fl_date", "route")
    .agg(
        F.count("*").alias("route_flights"),
        F.sum("is_delayed").alias("route_delayed")
    )
)

route_7d_window = (
    Window
    .partitionBy("route")
    .orderBy(F.unix_date(F.col("fl_date")))
    .rangeBetween(-7, -1)
)

route_daily_7d = (
    route_daily_7d
    .withColumn(
        "route_flights_last_7d",
        F.sum("route_flights").over(route_7d_window)
    )
    .withColumn(
        "route_delayed_last_7d",
        F.sum("route_delayed").over(route_7d_window)
    )
    .withColumn(
        "route_delay_rate_last_7d",
        F.when(
            F.col("route_flights_last_7d") > 0,
            F.col("route_delayed_last_7d") /
            F.col("route_flights_last_7d")
        )
    )
)

# COMMAND ----------

ml_df = ml_df.join(
    route_daily_7d.select(
        "fl_date",
        "route",
        "route_flights_last_7d",
        "route_delay_rate_last_7d"
    ),
    ["fl_date", "route"],
    "left"
)

# COMMAND ----------

origin_daily_7d = (
    ml_df
    .groupBy("fl_date", "origin")
    .agg(
        F.count("*").alias("origin_flights"),
        F.sum("is_delayed").alias("origin_delayed")
    )
)

origin_7d_window = (
    Window
    .partitionBy("origin")
    .orderBy(F.unix_date(F.col("fl_date")))
    .rangeBetween(-7, -1)
)

origin_daily_7d = (
    origin_daily_7d
    .withColumn(
        "origin_flights_last_7d",
        F.sum("origin_flights").over(origin_7d_window)
    )
    .withColumn(
        "origin_delayed_last_7d",
        F.sum("origin_delayed").over(origin_7d_window)
    )
    .withColumn(
        "origin_delay_rate_last_7d",
        F.when(
            F.col("origin_flights_last_7d") > 0,
            F.col("origin_delayed_last_7d") /
            F.col("origin_flights_last_7d")
        )
    )
)

ml_df = ml_df.join(
    origin_daily_7d.select(
        "fl_date",
        "origin",
        "origin_flights_last_7d",
        "origin_delay_rate_last_7d"
    ),
    ["fl_date", "origin"],
    "left"
)

# COMMAND ----------

carrier_daily_7d = (
    ml_df
    .groupBy("fl_date", "op_unique_carrier")
    .agg(
        F.count("*").alias("carrier_flights"),
        F.sum("is_delayed").alias("carrier_delayed")
    )
)

carrier_7d_window = (
    Window
    .partitionBy("op_unique_carrier")
    .orderBy(F.unix_date(F.col("fl_date")))
    .rangeBetween(-7, -1)
)

carrier_daily_7d = (
    carrier_daily_7d
    .withColumn(
        "carrier_flights_last_7d",
        F.sum("carrier_flights").over(carrier_7d_window)
    )
    .withColumn(
        "carrier_delayed_last_7d",
        F.sum("carrier_delayed").over(carrier_7d_window)
    )
    .withColumn(
        "carrier_delay_rate_last_7d",
        F.when(
            F.col("carrier_flights_last_7d") > 0,
            F.col("carrier_delayed_last_7d") /
            F.col("carrier_flights_last_7d")
        )
    )
)

ml_df = ml_df.join(
    carrier_daily_7d.select(
        "fl_date",
        "op_unique_carrier",
        "carrier_flights_last_7d",
        "carrier_delay_rate_last_7d"
    ),
    ["fl_date", "op_unique_carrier"],
    "left"
)

# COMMAND ----------

display(
    ml_df.select(
        "fl_date",
        "route",
        "route_delay_rate_last_7d",
        "origin",
        "origin_delay_rate_last_7d",
        "op_unique_carrier",
        "carrier_delay_rate_last_7d",
        "is_delayed"
    )
    .filter(F.col("fl_date") >= "2024-01-10")
    .limit(30)
)

# COMMAND ----------

feature_columns = [
    # Date / time
    "fl_date",
    "month",
    "day_of_week",
    "scheduled_dep_hour",
    "scheduled_dep_minute",
    "departure_period",
    "is_weekend",
    "dep_hour_sin",
    "dep_hour_cos",
    "month_sin",
    "month_cos",

    # Flight characteristics
    "op_unique_carrier",
    "origin",
    "dest",
    "route",
    "distance",

    # Long-term historical behavior
    "historical_route_flights",
    "reliable_route_delay_rate",
    "has_route_history",

    "historical_origin_flights",
    "reliable_origin_delay_rate",
    "has_origin_history",

    "historical_carrier_flights",
    "reliable_carrier_delay_rate",
    "has_carrier_history",

    # Recent operational behavior
    "route_flights_last_7d",
    "route_delay_rate_last_7d",

    "origin_flights_last_7d",
    "origin_delay_rate_last_7d",

    "carrier_flights_last_7d",
    "carrier_delay_rate_last_7d",

    # Target
    "is_delayed"
]

ml_features = ml_df.select(feature_columns)

print("Rows:", ml_features.count())
print("Columns:", len(ml_features.columns))


# COMMAND ----------

(
    ml_features.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.default.flight_ml_features")
)

# COMMAND ----------

display(
    spark.table("workspace.default.flight_ml_features").limit(10)
)

# COMMAND ----------

ml_df = ml_df.withColumn(
    "is_weekend",
    F.when(F.col("day_of_week").isin(6, 7), 1).otherwise(0)
)

# COMMAND ----------

display(
    ml_df.select(
        "fl_date",
        "day_of_week",
        "is_weekend"
    )
    .orderBy("fl_date")
    .limit(10)
)

# COMMAND ----------

display(
    ml_df.select(
        "fl_date",
        "day_of_week",
        "is_weekend"
    )
    .dropDuplicates()
    .orderBy("fl_date")
    .limit(10)
)

# COMMAND ----------

feature_columns = [
    "fl_date",
    "month",
    "day_of_week",
    "scheduled_dep_hour",
    "scheduled_dep_minute",
    "departure_period",
    "is_weekend",
    "dep_hour_sin",
    "dep_hour_cos",
    "month_sin",
    "month_cos",

    "op_unique_carrier",
    "origin",
    "dest",
    "route",
    "distance",

    "historical_route_flights",
    "reliable_route_delay_rate",
    "has_route_history",

    "historical_origin_flights",
    "reliable_origin_delay_rate",
    "has_origin_history",

    "historical_carrier_flights",
    "reliable_carrier_delay_rate",
    "has_carrier_history",

    "route_flights_last_7d",
    "route_delay_rate_last_7d",

    "origin_flights_last_7d",
    "origin_delay_rate_last_7d",

    "carrier_flights_last_7d",
    "carrier_delay_rate_last_7d",

    "is_delayed"
]

ml_features = ml_df.select(feature_columns)

print("Rows:", ml_features.count())
print("Columns:", len(ml_features.columns))

# COMMAND ----------

(
    ml_features.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.default.flight_ml_features")
)

# COMMAND ----------

display(
    spark.table("workspace.default.flight_ml_features").limit(10)
)

# COMMAND ----------

