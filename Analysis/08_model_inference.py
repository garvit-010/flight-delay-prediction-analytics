# Databricks notebook source
# ============================================================
# 1. IMPORTS + FEATURE CONFIGURATION
# ============================================================

import pandas as pd
import numpy as np
import pickle
import json
import joblib

from pyspark.sql import functions as F
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# Load feature columns used by the saved model
# ------------------------------------------------------------

with open("flight_delay_feature_cols.json", "r") as f:
    feature_cols = json.load(f)

TARGET_COL = "is_delayed"

print("Number of features:", len(feature_cols))
print("Target:", TARGET_COL)
print("\nFirst 10 features:")
print(feature_cols[:10])

# COMMAND ----------

# ============================================================
# 2. LOAD FEATURE TABLE
# ============================================================

df = spark.table(
    "workspace.default.flight_ml_features_v2"
)

print("Rows:", df.count())
print("Columns:", len(df.columns))

# COMMAND ----------

# ============================================================
# 3. RECREATE ORIGINAL TEMPORAL SPLITS
# ============================================================

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

# ============================================================
# 4. RECREATE EXACT TRAINING SAMPLE
# ============================================================

train_sample = train.sample(
    withReplacement=False,
    fraction=300000 / train.count(),
    seed=42
)

print("Training sample rows:", train_sample.count())

# COMMAND ----------

# ============================================================
# 5. CONVERT EXACT TRAINING SAMPLE TO PANDAS
# ============================================================

train_pd = train_sample.select(
    feature_cols + [TARGET_COL]
).toPandas()

print("Training Pandas shape:", train_pd.shape)
print("NaNs:", train_pd[feature_cols].isna().sum().sum())
print("Inf:", np.isinf(
    train_pd[feature_cols].select_dtypes(include=np.number)
).sum().sum())

# COMMAND ----------

# ============================================================
# 6. RECREATE ORIGINAL PREPROCESSING
# ============================================================

X_train_raw = train_pd[feature_cols].copy()
y_train = train_pd[TARGET_COL]

# ------------------------------------------------------------
# 1. MEDIAN IMPUTATION
# ------------------------------------------------------------

imputer = SimpleImputer(strategy="median")

X_train_imp = imputer.fit_transform(X_train_raw)

print("After imputation:")
print("Shape:", X_train_imp.shape)
print("NaNs:", np.isnan(X_train_imp).sum())

# ------------------------------------------------------------
# 2. STANDARD SCALING
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imp)

print("\nAfter scaling:")
print("Shape:", X_train_scaled.shape)
print("NaNs:", np.isnan(X_train_scaled).sum())
print("Inf:", np.isinf(X_train_scaled).sum())

# COMMAND ----------

# ============================================================
# 7. SAVE PREPROCESSING ARTIFACTS
# ============================================================

import joblib

joblib.dump(
    imputer,
    "flight_delay_imputer.pkl"
)

joblib.dump(
    scaler,
    "flight_delay_scaler.pkl"
)

print("Saved:")
print(" - flight_delay_imputer.pkl")
print(" - flight_delay_scaler.pkl")

# COMMAND ----------

# ============================================================
# 8. LOAD SAVED MODEL
# ============================================================

model = joblib.load("flight_delay_random_forest.pkl")

print("Model:", type(model).__name__)
print("Trees:", len(model.estimators_))
print("Features:", model.n_features_in_)

# COMMAND ----------

# ============================================================
# 9. PREPARE FULL TEST SET
# ============================================================

BUSINESS_COLS = [
    "fl_date",
    "op_unique_carrier",
    "origin",
    "dest",
    "route"
]

test_full_pd = (
    test
    .select(
        BUSINESS_COLS + feature_cols + [TARGET_COL]
    )
    .toPandas()
)

print("Test shape:", test_full_pd.shape)

print("\nTarget distribution:")
print(test_full_pd[TARGET_COL].value_counts())

print("\nTarget distribution (%):")
print(
    test_full_pd[TARGET_COL]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# COMMAND ----------

# ============================================================
# 10. APPLY SAVED PREPROCESSING
# ============================================================

# Load the exact preprocessing artifacts we just saved
imputer = joblib.load("flight_delay_imputer.pkl")
scaler = joblib.load("flight_delay_scaler.pkl")

# Raw test features
X_test_raw = test_full_pd[feature_cols].copy()

# 1. Imputation
X_test_imp = imputer.transform(X_test_raw)

# 2. Scaling
X_test_scaled = scaler.transform(X_test_imp)

print("Test scaled shape:", X_test_scaled.shape)
print("NaNs:", np.isnan(X_test_scaled).sum())
print("Inf:", np.isinf(X_test_scaled).sum())

# COMMAND ----------

# ============================================================
# 11. GENERATE TEST PREDICTIONS
# ============================================================

delay_probability = model.predict_proba(
    X_test_scaled
)[:, 1]

print("Probability distribution:")
print(
    pd.Series(delay_probability).describe(
        percentiles=[
            0.01, 0.05, 0.10,
            0.25, 0.50, 0.75,
            0.90, 0.95, 0.99
        ]
    )
)

print("\nProbability range:")
print("Min:", delay_probability.min())
print("Max:", delay_probability.max())
print("Mean:", delay_probability.mean())

# COMMAND ----------

# ============================================================
# 12. FINAL FULL TEST EVALUATION
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

VAL_THRESHOLD = 0.40

test_pred = (
    delay_probability >= VAL_THRESHOLD
).astype(int)

print("FINAL FULL TEST RESULTS")
print("=======================")

print("Threshold :", VAL_THRESHOLD)
print("Accuracy  :", accuracy_score(test_full_pd[TARGET_COL], test_pred))
print("Precision :", precision_score(test_full_pd[TARGET_COL], test_pred))
print("Recall    :", recall_score(test_full_pd[TARGET_COL], test_pred))
print("F1        :", f1_score(test_full_pd[TARGET_COL], test_pred))
print("ROC-AUC   :", roc_auc_score(
    test_full_pd[TARGET_COL],
    delay_probability
))
print("PR-AUC    :", average_precision_score(
    test_full_pd[TARGET_COL],
    delay_probability
))

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        test_full_pd[TARGET_COL],
        test_pred
    )
)

print("\nClassification Report:")
print(
    classification_report(
        test_full_pd[TARGET_COL],
        test_pred
    )
)

# COMMAND ----------

# ============================================================
# 13. CREATE FULL TEST PREDICTION DATASET
# ============================================================

prediction_df = test_full_pd[
    BUSINESS_COLS + [
        "scheduled_dep_hour",
        "distance",
        TARGET_COL
    ]
].copy()

# Model probability
prediction_df["delay_probability"] = delay_probability

# Binary prediction using validation-selected threshold
prediction_df["predicted_delay"] = (
    prediction_df["delay_probability"] >= VAL_THRESHOLD
).astype(int)

# Business-friendly risk category
prediction_df["risk_category"] = pd.cut(
    prediction_df["delay_probability"],
    bins=[-np.inf, 0.30, 0.50, np.inf],
    labels=["Low", "Medium", "High"]
)

print("Prediction dataset:", prediction_df.shape)

print("\nPredicted delay distribution:")
print(
    prediction_df["predicted_delay"]
    .value_counts()
)

print("\nRisk distribution:")
print(
    prediction_df["risk_category"]
    .value_counts()
)

display(prediction_df.head())

# COMMAND ----------

# ============================================================
# 14. SAVE PREDICTIONS TO DELTA TABLE
# ============================================================

prediction_spark_df = spark.createDataFrame(prediction_df)

prediction_spark_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(
        "workspace.default.flight_delay_predictions"
    )

print("Saved successfully.")
print(
    "Table: workspace.default.flight_delay_predictions"
)

# COMMAND ----------

# ============================================================
# VERIFY SAVED TABLE
# ============================================================

saved_predictions = spark.table(
    "workspace.default.flight_delay_predictions"
)

print("Rows:", saved_predictions.count())
print("Columns:", len(saved_predictions.columns))

display(saved_predictions.limit(10))

# COMMAND ----------

