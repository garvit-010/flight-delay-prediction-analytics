# Databricks notebook source
# ============================================================
# 1. LOAD MODEL PREDICTIONS
# ============================================================

from pyspark.sql import functions as F

predictions = spark.table(
    "workspace.default.flight_delay_predictions"
)

print("Rows:", predictions.count())
print("Columns:", len(predictions.columns))

display(predictions.limit(10))

# COMMAND ----------

# ============================================================
# 2. AIRLINE PERFORMANCE ANALYSIS
# ============================================================

airline_analysis = (
    predictions
    .groupBy("op_unique_carrier")
    .agg(
        F.count("*").alias("total_flights"),

        F.sum(
            F.col("is_delayed")
        ).alias("actual_delays"),

        F.sum(
            F.col("predicted_delay")
        ).alias("predicted_delays"),

        F.avg(
            F.col("is_delayed")
        ).alias("actual_delay_rate"),

        F.avg(
            F.col("predicted_delay")
        ).alias("predicted_delay_rate"),

        F.avg(
            F.col("delay_probability")
        ).alias("avg_risk")
    )
    .withColumn(
        "actual_delay_rate_pct",
        F.round(F.col("actual_delay_rate") * 100, 2)
    )
    .withColumn(
        "predicted_delay_rate_pct",
        F.round(F.col("predicted_delay_rate") * 100, 2)
    )
    .withColumn(
        "avg_risk_pct",
        F.round(F.col("avg_risk") * 100, 2)
    )
    .orderBy(
        F.desc("actual_delay_rate")
    )
)

display(airline_analysis)

# COMMAND ----------

# ============================================================
# 3. CARRIER RANKING — MINIMUM FLIGHT VOLUME
# ============================================================

from pyspark.sql.window import Window

carrier_ranking = (
    airline_analysis
    .filter(F.col("total_flights") >= 10000)
    .withColumn(
        "delay_rate_rank",
        F.row_number().over(
            Window.orderBy(F.desc("actual_delay_rate"))
        )
    )
    .select(
        "delay_rate_rank",
        "op_unique_carrier",
        "total_flights",
        "actual_delays",
        "actual_delay_rate_pct",
        "avg_risk_pct",
        "predicted_delay_rate_pct"
    )
)

display(carrier_ranking)

# COMMAND ----------

# ============================================================
# 4. ROUTE PERFORMANCE ANALYSIS
# ============================================================

route_analysis = (
    predictions
    .groupBy("route")
    .agg(
        F.count("*").alias("total_flights"),

        F.sum(
            F.col("is_delayed")
        ).alias("actual_delays"),

        F.sum(
            F.col("predicted_delay")
        ).alias("predicted_delays"),

        F.avg(
            F.col("is_delayed")
        ).alias("actual_delay_rate"),

        F.avg(
            F.col("delay_probability")
        ).alias("avg_risk")
    )
    .withColumn(
        "actual_delay_rate_pct",
        F.round(F.col("actual_delay_rate") * 100, 2)
    )
    .withColumn(
        "avg_risk_pct",
        F.round(F.col("avg_risk") * 100, 2)
    )
    .filter(
        F.col("total_flights") >= 500
    )
    .orderBy(
        F.desc("actual_delay_rate")
    )
)

display(route_analysis.limit(20))

# COMMAND ----------

# ============================================================
# 5. DEPARTURE HOUR ANALYSIS
# ============================================================

hour_analysis = (
    predictions
    .groupBy("scheduled_dep_hour")
    .agg(
        F.count("*").alias("total_flights"),

        F.sum(
            F.col("is_delayed")
        ).alias("actual_delays"),

        F.sum(
            F.col("predicted_delay")
        ).alias("predicted_delays"),

        F.avg(
            F.col("is_delayed")
        ).alias("actual_delay_rate"),

        F.avg(
            F.col("delay_probability")
        ).alias("avg_risk")
    )
    .withColumn(
        "actual_delay_rate_pct",
        F.round(F.col("actual_delay_rate") * 100, 2)
    )
    .withColumn(
        "avg_risk_pct",
        F.round(F.col("avg_risk") * 100, 2)
    )
    .orderBy("scheduled_dep_hour")
)

display(hour_analysis)

# COMMAND ----------

# ============================================================
# 6. RISK CATEGORY ANALYSIS
# ============================================================

risk_analysis = (
    predictions
    .groupBy("risk_category")
    .agg(
        F.count("*").alias("total_flights"),

        F.sum(
            F.col("is_delayed")
        ).alias("actual_delays"),

        F.sum(
            F.col("predicted_delay")
        ).alias("predicted_delays"),

        F.avg(
            F.col("is_delayed")
        ).alias("actual_delay_rate"),

        F.avg(
            F.col("delay_probability")
        ).alias("avg_risk")
    )
    .withColumn(
        "actual_delay_rate_pct",
        F.round(F.col("actual_delay_rate") * 100, 2)
    )
    .withColumn(
        "avg_risk_pct",
        F.round(F.col("avg_risk") * 100, 2)
    )
)

display(risk_analysis)

# COMMAND ----------

# ============================================================
# 7. HIGH-RISK OPERATIONAL COVERAGE
# ============================================================

total_actual_delays = (
    predictions
    .agg(F.sum("is_delayed").alias("total_delays"))
    .collect()[0]["total_delays"]
)

high_risk_delays = (
    predictions
    .filter(F.col("risk_category") == "High")
    .agg(F.sum("is_delayed").alias("delays"))
    .collect()[0]["delays"]
)

high_risk_flights = (
    predictions
    .filter(F.col("risk_category") == "High")
    .count()
)

total_flights = predictions.count()

print("Total flights:", total_flights)
print("Total actual delays:", total_actual_delays)
print("High-risk flights:", high_risk_flights)
print("Actual delays within high-risk:", high_risk_delays)

print(
    "\nHigh-risk share of flights:",
    round(high_risk_flights / total_flights * 100, 2),
    "%"
)

print(
    "High-risk share of actual delays:",
    round(high_risk_delays / total_actual_delays * 100, 2),
    "%"
)

# COMMAND ----------

# ============================================================
# 8. MODEL ERROR ANALYSIS
# ============================================================

error_analysis = (
    predictions
    .withColumn(
        "prediction_type",
        F.when(
            (F.col("predicted_delay") == 1) &
            (F.col("is_delayed") == 1),
            "True Positive"
        )
        .when(
            (F.col("predicted_delay") == 1) &
            (F.col("is_delayed") == 0),
            "False Positive"
        )
        .when(
            (F.col("predicted_delay") == 0) &
            (F.col("is_delayed") == 0),
            "True Negative"
        )
        .otherwise("False Negative")
    )
    .groupBy("prediction_type")
    .agg(
        F.count("*").alias("flights")
    )
)

display(error_analysis)

# COMMAND ----------

# ============================================================
# 9. CARRIER-LEVEL MODEL PERFORMANCE
# ============================================================

carrier_model_analysis = (
    predictions
    .groupBy("op_unique_carrier")
    .agg(
        F.count("*").alias("total_flights"),

        F.sum("is_delayed").alias("actual_delays"),

        F.sum("predicted_delay").alias("predicted_delays"),

        # True positives
        F.sum(
            F.when(
                (F.col("predicted_delay") == 1) &
                (F.col("is_delayed") == 1),
                1
            ).otherwise(0)
        ).alias("true_positives"),

        # False positives
        F.sum(
            F.when(
                (F.col("predicted_delay") == 1) &
                (F.col("is_delayed") == 0),
                1
            ).otherwise(0)
        ).alias("false_positives"),

        # False negatives
        F.sum(
            F.when(
                (F.col("predicted_delay") == 0) &
                (F.col("is_delayed") == 1),
                1
            ).otherwise(0)
        ).alias("false_negatives"),

        F.avg("delay_probability").alias("avg_risk")
    )
    .withColumn(
        "actual_delay_rate_pct",
        F.round(
            F.col("actual_delays") /
            F.col("total_flights") * 100,
            2
        )
    )
    .withColumn(
        "precision_pct",
        F.round(
            F.col("true_positives") /
            F.col("predicted_delays") * 100,
            2
        )
    )
    .withColumn(
        "recall_pct",
        F.round(
            F.col("true_positives") /
            F.col("actual_delays") * 100,
            2
        )
    )
    .withColumn(
        "avg_risk_pct",
        F.round(F.col("avg_risk") * 100, 2)
    )
    .filter(
        F.col("total_flights") >= 10000
    )
    .orderBy(
        F.desc("actual_delay_rate_pct")
    )
)

display(carrier_model_analysis)

# COMMAND ----------

# ============================================================
# 10. DAILY DELAY TREND
# ============================================================

daily_analysis = (
    predictions
    .groupBy("fl_date")
    .agg(
        F.count("*").alias("total_flights"),

        F.sum("is_delayed").alias("actual_delays"),

        F.sum("predicted_delay").alias("predicted_delays"),

        F.avg("is_delayed").alias("actual_delay_rate"),

        F.avg("delay_probability").alias("avg_risk")
    )
    .withColumn(
        "actual_delay_rate_pct",
        F.round(F.col("actual_delay_rate") * 100, 2)
    )
    .withColumn(
        "avg_risk_pct",
        F.round(F.col("avg_risk") * 100, 2)
    )
    .orderBy("fl_date")
)

display(daily_analysis)

# COMMAND ----------

# ============================================================
# CELL 11 — SAVE DASHBOARD-READY AGGREGATE TABLES
# ============================================================

# 1. Airline analysis
airline_analysis.write.mode("overwrite").saveAsTable(
    "workspace.default.flight_delay_airline_analysis"
)

# 2. Route analysis
route_analysis.write.mode("overwrite").saveAsTable(
    "workspace.default.flight_delay_route_analysis"
)

# 3. Departure-hour analysis
hour_analysis.write.mode("overwrite").saveAsTable(
    "workspace.default.flight_delay_hour_analysis"
)

# 4. Risk-category analysis
risk_analysis.write.mode("overwrite").saveAsTable(
    "workspace.default.flight_delay_risk_analysis"
)

# 5. Daily trend
daily_analysis.write.mode("overwrite").saveAsTable(
    "workspace.default.flight_delay_daily_analysis"
)

print("All dashboard aggregate tables saved successfully.")

# COMMAND ----------

# ============================================================
# CELL 12 — VERIFY DASHBOARD TABLES
# ============================================================

tables = [
    "flight_delay_airline_analysis",
    "flight_delay_route_analysis",
    "flight_delay_hour_analysis",
    "flight_delay_risk_analysis",
    "flight_delay_daily_analysis"
]

for table in tables:
    df_check = spark.table(f"workspace.default.{table}")
    
    print(f"{table}: {df_check.count():,} rows × {len(df_check.columns)} columns")

# COMMAND ----------

# ============================================================
# CELL 13 — MODEL KPI SUMMARY
# ============================================================

model_kpis = spark.createDataFrame([
    ("Total Flights", 584832.0),
    ("Actual Delays", 123721.0),
    ("Actual Delay Rate", 0.2115),
    ("Accuracy", 0.5935208744),
    ("Precision", 0.28981419),
    ("Recall", 0.63525998),
    ("F1 Score", 0.39803804),
    ("ROC-AUC", 0.6522508663),
    ("PR-AUC", 0.3315248270),
    ("High-Risk Flights", 124362.0),
    ("High-Risk Delay Capture", 0.3529),
], ["metric", "value"])

display(model_kpis)

# COMMAND ----------

model_kpis.write.mode("overwrite").saveAsTable(
    "workspace.default.flight_delay_model_kpis"
)

print("Model KPI table saved successfully.")

# COMMAND ----------

