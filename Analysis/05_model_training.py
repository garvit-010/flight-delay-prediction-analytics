# Databricks notebook source
from pyspark.sql import functions as F

df = spark.table("workspace.default.flight_ml_features")

print("Rows:", df.count())
print("Columns:", len(df.columns))

# COMMAND ----------

train = df.filter(
    F.col("fl_date") < "2024-10-01"
)

validation = df.filter(
    (F.col("fl_date") >= "2024-10-01") &
    (F.col("fl_date") < "2024-12-01")
)

test = df.filter(
    F.col("fl_date") >= "2024-12-01"
)

print("Train:", train.count())
print("Validation:", validation.count())
print("Test:", test.count())

# COMMAND ----------

display(
    train.groupBy("is_delayed")
    .count()
    .withColumn(
        "percentage",
        F.round(
            F.col("count") / train.count() * 100,
            2
        )
    )
)

# COMMAND ----------

display(
    validation.groupBy("is_delayed")
    .count()
    .withColumn(
        "percentage",
        F.round(
            F.col("count") / validation.count() * 100,
            2
        )
    )
)

# COMMAND ----------

display(
    test.groupBy("is_delayed")
    .count()
    .withColumn(
        "percentage",
        F.round(
            F.col("count") / test.count() * 100,
            2
        )
    )
)

# COMMAND ----------

feature_cols = [
    # Time
    "month",
    "day_of_week",
    "scheduled_dep_hour",
    "scheduled_dep_minute",
    "is_weekend",
    "dep_hour_sin",
    "dep_hour_cos",
    "month_sin",
    "month_cos",

    # Flight
    "distance",

    # Frequency encoding
    "op_unique_carrier_frequency",
    "origin_frequency",
    "dest_frequency",
    "route_frequency",

    # Long-term historical
    "historical_route_flights",
    "reliable_route_delay_rate",
    "has_route_history",

    "historical_origin_flights",
    "reliable_origin_delay_rate",
    "has_origin_history",

    "historical_carrier_flights",
    "reliable_carrier_delay_rate",
    "has_carrier_history",

    # Recent 7-day history
    "route_flights_last_7d",
    "route_delay_rate_last_7d",

    "origin_flights_last_7d",
    "origin_delay_rate_last_7d",

    "carrier_flights_last_7d",
    "carrier_delay_rate_last_7d"
]

# COMMAND ----------

categorical_cols = [
    "op_unique_carrier",
    "origin",
    "dest",
    "route"
]

for col in categorical_cols:

    freq = (
        train
        .groupBy(col)
        .count()
        .withColumnRenamed(
            "count",
            f"{col}_frequency"
        )
    )

    train = train.join(
        freq,
        on=col,
        how="left"
    )

    validation = validation.join(
        freq,
        on=col,
        how="left"
    )

    test = test.join(
        freq,
        on=col,
        how="left"
    )

# COMMAND ----------

display(
    train.select(
        "op_unique_carrier",
        "op_unique_carrier_frequency",
        "origin",
        "origin_frequency",
        "dest",
        "dest_frequency",
        "route",
        "route_frequency"
    ).limit(20)
)

# COMMAND ----------

train_sample = (
    train
    .sample(
        withReplacement=False,
        fraction=300000 / train.count(),
        seed=42
    )
)

# COMMAND ----------

print("Sample rows:", train_sample.count())

display(
    train_sample.groupBy("is_delayed")
    .count()
    .withColumn(
        "percentage",
        F.round(
            F.col("count") / train_sample.count() * 100,
            2
        )
    )
)

# COMMAND ----------

validation_sample = validation.sample(
    fraction=0.20,
    seed=42
)

test_sample = test.sample(
    fraction=0.30,
    seed=42
)

# COMMAND ----------

train_pd = train_sample.select(
    feature_cols + ["is_delayed"]
).toPandas()

validation_pd = validation_sample.select(
    feature_cols + ["is_delayed"]
).toPandas()

test_pd = test_sample.select(
    feature_cols + ["is_delayed"]
).toPandas()

# COMMAND ----------

print("Train:", train_pd.shape)
print("Validation:", validation_pd.shape)
print("Test:", test_pd.shape)

# COMMAND ----------

import numpy as np

print("NaN values:")
print(train_pd.isna().sum().sum())

print("Infinite values:")
print(np.isinf(train_pd.select_dtypes(include=np.number)).sum().sum())

# COMMAND ----------

nan_counts = train_pd[feature_cols].isna().sum()

display(
    nan_counts[nan_counts > 0]
    .sort_values(ascending=False)
)

# COMMAND ----------

from sklearn.impute import SimpleImputer
import numpy as np

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(
    train_pd[feature_cols]
)

X_val = imputer.transform(
    validation_pd[feature_cols]
)

X_test = imputer.transform(
    test_pd[feature_cols]
)

# COMMAND ----------

y_train = train_pd["is_delayed"].values
y_val = validation_pd["is_delayed"].values
y_test = test_pd["is_delayed"].values

# COMMAND ----------

print("Train NaNs:", np.isnan(X_train).sum())
print("Validation NaNs:", np.isnan(X_val).sum())
print("Test NaNs:", np.isnan(X_test).sum())

# COMMAND ----------

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

baseline_pred = np.zeros(len(y_val))

print("BASELINE")
print("Accuracy :", accuracy_score(y_val, baseline_pred))
print("Precision:", precision_score(y_val, baseline_pred, zero_division=0))
print("Recall   :", recall_score(y_val, baseline_pred, zero_division=0))
print("F1       :", f1_score(y_val, baseline_pred, zero_division=0))

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs"
)

lr.fit(X_train_scaled, y_train)

# COMMAND ----------

lr_pred = lr.predict(X_val_scaled)
lr_prob = lr.predict_proba(X_val_scaled)[:, 1]

# COMMAND ----------

print("LOGISTIC REGRESSION")
print("-------------------")
print("Accuracy :", accuracy_score(y_val, lr_pred))
print("Precision:", precision_score(y_val, lr_pred))
print("Recall   :", recall_score(y_val, lr_pred))
print("F1       :", f1_score(y_val, lr_pred))
print("ROC-AUC  :", roc_auc_score(y_val, lr_prob))
print("PR-AUC   :", average_precision_score(y_val, lr_prob))

# COMMAND ----------

print("Confusion Matrix:")
print(confusion_matrix(y_val, lr_pred))

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_leaf=20,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

# COMMAND ----------

rf.fit(X_train, y_train)

# COMMAND ----------

rf_pred = rf.predict(X_val)
rf_prob = rf.predict_proba(X_val)[:, 1]

# COMMAND ----------

print("RANDOM FOREST")
print("-------------")
print("Accuracy :", accuracy_score(y_val, rf_pred))
print("Precision:", precision_score(y_val, rf_pred))
print("Recall   :", recall_score(y_val, rf_pred))
print("F1       :", f1_score(y_val, rf_pred))
print("ROC-AUC  :", roc_auc_score(y_val, rf_prob))
print("PR-AUC   :", average_precision_score(y_val, rf_prob))

# COMMAND ----------

print(confusion_matrix(y_val, rf_pred))

# COMMAND ----------

import pandas as pd

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

display(importance.head(20))

# COMMAND ----------

thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

for threshold in thresholds:

    pred = (rf_prob >= threshold).astype(int)

    print(
        f"Threshold {threshold:.2f} | "
        f"Precision: {precision_score(y_val, pred):.3f} | "
        f"Recall: {recall_score(y_val, pred):.3f} | "
        f"F1: {f1_score(y_val, pred):.3f}"
    )

# COMMAND ----------

import sys
import platform

print("Python:", sys.version)
print("Platform:", platform.platform())

# COMMAND ----------

# MAGIC %pip show scikit-learn

# COMMAND ----------

# MAGIC %pip install xgboost-cpu==3.3.0

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import xgboost as xgb

print(xgb.__version__)

# COMMAND ----------

from pyspark.sql import functions as F

df = spark.table("workspace.default.flight_ml_features")

train = df.filter(F.col("fl_date") < "2024-10-01")

validation = df.filter(
    (F.col("fl_date") >= "2024-10-01") &
    (F.col("fl_date") < "2024-12-01")
)

test = df.filter(F.col("fl_date") >= "2024-12-01")

print("Train:", train.count())
print("Validation:", validation.count())
print("Test:", test.count())

# COMMAND ----------

categorical_cols = [
    "op_unique_carrier",
    "origin",
    "dest",
    "route"
]

for col in categorical_cols:

    freq = (
        train
        .groupBy(col)
        .count()
        .withColumnRenamed(
            "count",
            f"{col}_frequency"
        )
    )

    train = train.join(freq, on=col, how="left")
    validation = validation.join(freq, on=col, how="left")
    test = test.join(freq, on=col, how="left")

# COMMAND ----------

feature_cols = [
    "month",
    "day_of_week",
    "scheduled_dep_hour",
    "scheduled_dep_minute",
    "is_weekend",

    "dep_hour_sin",
    "dep_hour_cos",
    "month_sin",
    "month_cos",

    "distance",

    "op_unique_carrier_frequency",
    "origin_frequency",
    "dest_frequency",
    "route_frequency",

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
    "carrier_delay_rate_last_7d"
]

# COMMAND ----------

train_sample = train.sample(
    withReplacement=False,
    fraction=300000 / train.count(),
    seed=42
)

validation_sample = validation.sample(
    fraction=0.20,
    seed=42
)

test_sample = test.sample(
    fraction=0.30,
    seed=42
)

print("Train sample:", train_sample.count())
print("Validation sample:", validation_sample.count())
print("Test sample:", test_sample.count())

# COMMAND ----------

train_pd = train_sample.select(
    feature_cols + ["is_delayed"]
).toPandas()

validation_pd = validation_sample.select(
    feature_cols + ["is_delayed"]
).toPandas()

test_pd = test_sample.select(
    feature_cols + ["is_delayed"]
).toPandas()

print("Train:", train_pd.shape)
print("Validation:", validation_pd.shape)
print("Test:", test_pd.shape)

# COMMAND ----------

from sklearn.impute import SimpleImputer
import numpy as np

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(
    train_pd[feature_cols]
)

X_val = imputer.transform(
    validation_pd[feature_cols]
)

X_test = imputer.transform(
    test_pd[feature_cols]
)

y_train = train_pd["is_delayed"].values
y_val = validation_pd["is_delayed"].values
y_test = test_pd["is_delayed"].values

print("Train NaNs:", np.isnan(X_train).sum())
print("Validation NaNs:", np.isnan(X_val).sum())
print("Test NaNs:", np.isnan(X_test).sum())

# COMMAND ----------

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("Negative:", negative)
print("Positive:", positive)
print("Scale positive weight:", scale_pos_weight)

# COMMAND ----------

from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train,
    y_train
)

# COMMAND ----------

xgb_prob = xgb_model.predict_proba(X_val)[:, 1]
xgb_pred = (xgb_prob >= 0.5).astype(int)

# COMMAND ----------

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# COMMAND ----------

print("XGBOOST — VALIDATION")
print("--------------------")
print("Accuracy :", accuracy_score(y_val, xgb_pred))
print("Precision:", precision_score(y_val, xgb_pred))
print("Recall   :", recall_score(y_val, xgb_pred))
print("F1       :", f1_score(y_val, xgb_pred))
print("ROC-AUC  :", roc_auc_score(y_val, xgb_prob))
print("PR-AUC   :", average_precision_score(y_val, xgb_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, xgb_pred))

# COMMAND ----------

thresholds = [
    0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50
]

for threshold in thresholds:
    pred = (xgb_prob >= threshold).astype(int)

    print(
        f"Threshold {threshold:.2f} | "
        f"Precision: {precision_score(y_val, pred):.3f} | "
        f"Recall: {recall_score(y_val, pred):.3f} | "
        f"F1: {f1_score(y_val, pred):.3f}"
    )

# COMMAND ----------

