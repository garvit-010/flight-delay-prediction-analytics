# Databricks notebook source
df = spark.table("workspace.default.flight_ml_features")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

df = spark.table("workspace.default.flight_ml_features")

print("Rows:", df.count())
print("Columns:", len(df.columns))

# COMMAND ----------

df_v2 = (
    df
    .withColumn(
        "origin_hour_key",
        F.concat_ws("_", "origin", F.col("scheduled_dep_hour").cast("string"))
    )
    .withColumn(
        "route_hour_key",
        F.concat_ws("_", "route", F.col("scheduled_dep_hour").cast("string"))
    )
    .withColumn(
        "carrier_hour_key",
        F.concat_ws("_", "op_unique_carrier", F.col("scheduled_dep_hour").cast("string"))
    )
    .withColumn(
        "origin_dow_key",
        F.concat_ws("_", "origin", F.col("day_of_week").cast("string"))
    )
    .withColumn(
        "route_dow_key",
        F.concat_ws("_", "route", F.col("day_of_week").cast("string"))
    )
)

# COMMAND ----------

display(
    df_v2.select(
        "fl_date",
        "origin",
        "route",
        "op_unique_carrier",
        "scheduled_dep_hour",
        "day_of_week",
        "origin_hour_key",
        "route_hour_key",
        "carrier_hour_key",
        "origin_dow_key",
        "route_dow_key"
    ).limit(20)
)

# COMMAND ----------

df_v2 = df_v2.withColumn(
    "scheduled_dep_minutes",
    F.col("scheduled_dep_hour") * 60
    + F.col("scheduled_dep_minute")
)

# COMMAND ----------

df_v2 = df_v2.withColumn(
    "flight_datetime_order",
    F.to_timestamp(
        F.concat_ws(
            " ",
            F.col("fl_date").cast("string"),
            F.format_string(
                "%02d:%02d:00",
                F.col("scheduled_dep_hour"),
                F.col("scheduled_dep_minute")
            )
        )
    )
)

# COMMAND ----------

display(
    df_v2.select(
        "fl_date",
        "scheduled_dep_hour",
        "scheduled_dep_minute",
        "flight_datetime_order"
    )
    .orderBy("flight_datetime_order")
    .limit(20)
)

# COMMAND ----------

df_v2 = df_v2.withColumn(
    "flight_timestamp",
    F.col("flight_datetime_order").cast("long")
)

# COMMAND ----------

display(
    df_v2.select(
        "fl_date",
        "scheduled_dep_hour",
        "scheduled_dep_minute",
        "flight_datetime_order",
        "flight_timestamp"
    )
    .orderBy("flight_datetime_order")
    .limit(10)
)

# COMMAND ----------

origin_hour_window = (
    Window
    .partitionBy("origin_hour_key")
    .orderBy("flight_timestamp")
    .rangeBetween(
        Window.unboundedPreceding,
        -1
    )
)

# COMMAND ----------

df_v2 = (
    df_v2
    .withColumn(
        "historical_origin_hour_flights",
        F.count("*").over(origin_hour_window)
    )
    .withColumn(
        "historical_origin_hour_delayed",
        F.sum("is_delayed").over(origin_hour_window)
    )
    .withColumn(
        "historical_origin_hour_delay_rate",
        F.when(
            F.col("historical_origin_hour_flights") > 0,
            F.col("historical_origin_hour_delayed")
            / F.col("historical_origin_hour_flights")
        )
    )
)

# COMMAND ----------

display(
    df_v2
    .filter(F.col("origin_hour_key") == "JFK_12")
    .select(
        "fl_date",
        "flight_datetime_order",
        "origin",
        "scheduled_dep_hour",
        "is_delayed",
        "historical_origin_hour_flights",
        "historical_origin_hour_delayed",
        "historical_origin_hour_delay_rate"
    )
    .orderBy("flight_datetime_order")
    .limit(30)
)

# COMMAND ----------

route_hour_window = (
    Window
    .partitionBy("route_hour_key")
    .orderBy("flight_timestamp")
    .rangeBetween(
        Window.unboundedPreceding,
        -1
    )
)

df_v2 = (
    df_v2
    .withColumn(
        "historical_route_hour_flights",
        F.count("*").over(route_hour_window)
    )
    .withColumn(
        "historical_route_hour_delayed",
        F.sum("is_delayed").over(route_hour_window)
    )
    .withColumn(
        "historical_route_hour_delay_rate",
        F.when(
            F.col("historical_route_hour_flights") > 0,
            F.col("historical_route_hour_delayed")
            / F.col("historical_route_hour_flights")
        )
    )
)

# COMMAND ----------

carrier_hour_window = (
    Window
    .partitionBy("carrier_hour_key")
    .orderBy("flight_timestamp")
    .rangeBetween(
        Window.unboundedPreceding,
        -1
    )
)

df_v2 = (
    df_v2
    .withColumn(
        "historical_carrier_hour_flights",
        F.count("*").over(carrier_hour_window)
    )
    .withColumn(
        "historical_carrier_hour_delayed",
        F.sum("is_delayed").over(carrier_hour_window)
    )
    .withColumn(
        "historical_carrier_hour_delay_rate",
        F.when(
            F.col("historical_carrier_hour_flights") > 0,
            F.col("historical_carrier_hour_delayed")
            / F.col("historical_carrier_hour_flights")
        )
    )
)

# COMMAND ----------

origin_dow_window = (
    Window
    .partitionBy("origin_dow_key")
    .orderBy("flight_timestamp")
    .rangeBetween(
        Window.unboundedPreceding,
        -1
    )
)

df_v2 = (
    df_v2
    .withColumn(
        "historical_origin_dow_flights",
        F.count("*").over(origin_dow_window)
    )
    .withColumn(
        "historical_origin_dow_delayed",
        F.sum("is_delayed").over(origin_dow_window)
    )
    .withColumn(
        "historical_origin_dow_delay_rate",
        F.when(
            F.col("historical_origin_dow_flights") > 0,
            F.col("historical_origin_dow_delayed")
            / F.col("historical_origin_dow_flights")
        )
    )
)

# COMMAND ----------

route_dow_window = (
    Window
    .partitionBy("route_dow_key")
    .orderBy("flight_timestamp")
    .rangeBetween(
        Window.unboundedPreceding,
        -1
    )
)

df_v2 = (
    df_v2
    .withColumn(
        "historical_route_dow_flights",
        F.count("*").over(route_dow_window)
    )
    .withColumn(
        "historical_route_dow_delayed",
        F.sum("is_delayed").over(route_dow_window)
    )
    .withColumn(
        "historical_route_dow_delay_rate",
        F.when(
            F.col("historical_route_dow_flights") > 0,
            F.col("historical_route_dow_delayed")
            / F.col("historical_route_dow_flights")
        )
    )
)

# COMMAND ----------

df_v2 = (
    df_v2
    .withColumn(
        "reliable_origin_hour_delay_rate",
        F.when(
            F.col("historical_origin_hour_flights") >= 20,
            F.col("historical_origin_hour_delay_rate")
        )
    )
    .withColumn(
        "reliable_route_hour_delay_rate",
        F.when(
            F.col("historical_route_hour_flights") >= 20,
            F.col("historical_route_hour_delay_rate")
        )
    )
    .withColumn(
        "reliable_carrier_hour_delay_rate",
        F.when(
            F.col("historical_carrier_hour_flights") >= 20,
            F.col("historical_carrier_hour_delay_rate")
        )
    )
    .withColumn(
        "reliable_origin_dow_delay_rate",
        F.when(
            F.col("historical_origin_dow_flights") >= 20,
            F.col("historical_origin_dow_delay_rate")
        )
    )
    .withColumn(
        "reliable_route_dow_delay_rate",
        F.when(
            F.col("historical_route_dow_flights") >= 20,
            F.col("historical_route_dow_delay_rate")
        )
    )
)

# COMMAND ----------

new_features = [
    "historical_origin_hour_flights",
    "reliable_origin_hour_delay_rate",
    "historical_route_hour_flights",
    "reliable_route_hour_delay_rate",
    "historical_carrier_hour_flights",
    "reliable_carrier_hour_delay_rate",
    "historical_origin_dow_flights",
    "reliable_origin_dow_delay_rate",
    "historical_route_dow_flights",
    "reliable_route_dow_delay_rate"
]

display(
    df_v2.select(
        "fl_date",
        "origin",
        "route",
        "op_unique_carrier",
        "scheduled_dep_hour",
        "day_of_week",
        *new_features,
        "is_delayed"
    )
    .filter(F.col("fl_date") >= "2024-02-01")
    .limit(30)
)

# COMMAND ----------

origin_hour_7d_window = (
    Window
    .partitionBy("origin_hour_key")
    .orderBy("flight_timestamp")
    .rangeBetween(
        -7 * 24 * 60 * 60,
        -1
    )
)

# COMMAND ----------

df_v2 = (
    df_v2
    .withColumn(
        "origin_hour_flights_last_7d",
        F.count("*").over(origin_hour_7d_window)
    )
    .withColumn(
        "origin_hour_delayed_last_7d",
        F.sum("is_delayed").over(origin_hour_7d_window)
    )
    .withColumn(
        "origin_hour_delay_rate_last_7d",
        F.when(
            F.col("origin_hour_flights_last_7d") > 0,
            F.col("origin_hour_delayed_last_7d")
            / F.col("origin_hour_flights_last_7d")
        )
    )
)

# COMMAND ----------

display(
    df_v2
    .filter(
        (F.col("origin") == "ABE") &
        (F.col("scheduled_dep_hour") == 6)
    )
    .select(
        "fl_date",
        "flight_datetime_order",
        "origin",
        "scheduled_dep_hour",
        "is_delayed",
        "origin_hour_flights_last_7d",
        "origin_hour_delay_rate_last_7d"
    )
    .orderBy("flight_datetime_order")
    .limit(30)
)

# COMMAND ----------

route_hour_7d_window = (
    Window
    .partitionBy("route_hour_key")
    .orderBy("flight_timestamp")
    .rangeBetween(
        -7 * 24 * 60 * 60,
        -1
    )
)

df_v2 = (
    df_v2
    .withColumn(
        "route_hour_flights_last_7d",
        F.count("*").over(route_hour_7d_window)
    )
    .withColumn(
        "route_hour_delayed_last_7d",
        F.sum("is_delayed").over(route_hour_7d_window)
    )
    .withColumn(
        "route_hour_delay_rate_last_7d",
        F.when(
            F.col("route_hour_flights_last_7d") > 0,
            F.col("route_hour_delayed_last_7d")
            / F.col("route_hour_flights_last_7d")
        )
    )
)

# COMMAND ----------

carrier_hour_7d_window = (
    Window
    .partitionBy("carrier_hour_key")
    .orderBy("flight_timestamp")
    .rangeBetween(
        -7 * 24 * 60 * 60,
        -1
    )
)

df_v2 = (
    df_v2
    .withColumn(
        "carrier_hour_flights_last_7d",
        F.count("*").over(carrier_hour_7d_window)
    )
    .withColumn(
        "carrier_hour_delayed_last_7d",
        F.sum("is_delayed").over(carrier_hour_7d_window)
    )
    .withColumn(
        "carrier_hour_delay_rate_last_7d",
        F.when(
            F.col("carrier_hour_flights_last_7d") > 0,
            F.col("carrier_hour_delayed_last_7d")
            / F.col("carrier_hour_flights_last_7d")
        )
    )
)

# COMMAND ----------

display(
    df_v2.select(
        "fl_date",
        "origin",
        "route",
        "op_unique_carrier",
        "scheduled_dep_hour",
        "origin_hour_flights_last_7d",
        "origin_hour_delay_rate_last_7d",
        "route_hour_flights_last_7d",
        "route_hour_delay_rate_last_7d",
        "carrier_hour_flights_last_7d",
        "carrier_hour_delay_rate_last_7d",
        "is_delayed"
    )
    .filter(F.col("fl_date") >= "2024-01-08")
    .limit(30)
)

# COMMAND ----------

drop_cols = [
    "origin_hour_key",
    "route_hour_key",
    "carrier_hour_key",
    "origin_dow_key",
    "route_dow_key",
    "scheduled_dep_minutes",
    "flight_datetime_order",
    "flight_timestamp",

    "historical_origin_hour_delayed",
    "historical_route_hour_delayed",
    "historical_carrier_hour_delayed",
    "historical_origin_dow_delayed",
    "historical_route_dow_delayed",

    "origin_hour_delayed_last_7d",
    "route_hour_delayed_last_7d",
    "carrier_hour_delayed_last_7d"
]

df_v2_final = df_v2.drop(*drop_cols)

# COMMAND ----------

(
    df_v2_final
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("workspace.default.flight_ml_features_v2")
)

# COMMAND ----------

print(
    "Rows:",
    spark.table("workspace.default.flight_ml_features_v2").count()
)

print(
    "Columns:",
    len(
        spark.table(
            "workspace.default.flight_ml_features_v2"
        ).columns
    )
)

# COMMAND ----------

v2 = spark.table("workspace.default.flight_ml_features_v2")

display(
    v2.select(
        "fl_date",
        "origin",
        "route",
        "scheduled_dep_hour",

        "reliable_origin_hour_delay_rate",
        "reliable_route_hour_delay_rate",
        "reliable_carrier_hour_delay_rate",

        "reliable_origin_dow_delay_rate",
        "reliable_route_dow_delay_rate",

        "origin_hour_delay_rate_last_7d",
        "route_hour_delay_rate_last_7d",
        "carrier_hour_delay_rate_last_7d",

        "is_delayed"
    )
    .filter(F.col("fl_date") >= "2024-02-01")
    .limit(20)
)

# COMMAND ----------

