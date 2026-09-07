# Databricks notebook source
# MAGIC %restart_python

# COMMAND ----------

from pyspark.sql import functions as F

df = spark.table(
    "workspace.default.flight_ml_features_v2"
)

print("Rows:", df.count())
print("Columns:", len(df.columns))

# COMMAND ----------

print(df.columns)

# COMMAND ----------

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

    # Existing historical features
    "historical_route_flights",
    "reliable_route_delay_rate",
    "has_route_history",

    "historical_origin_flights",
    "reliable_origin_delay_rate",
    "has_origin_history",

    "historical_carrier_flights",
    "reliable_carrier_delay_rate",
    "has_carrier_history",

    # Existing rolling features
    "route_flights_last_7d",
    "route_delay_rate_last_7d",

    "origin_flights_last_7d",
    "origin_delay_rate_last_7d",

    "carrier_flights_last_7d",
    "carrier_delay_rate_last_7d",

    # V2: historical context
    "historical_origin_hour_flights",
    "historical_origin_hour_delay_rate",

    "historical_route_hour_flights",
    "historical_route_hour_delay_rate",

    "historical_carrier_hour_flights",
    "historical_carrier_hour_delay_rate",

    "historical_origin_dow_flights",
    "historical_origin_dow_delay_rate",

    "historical_route_dow_flights",
    "historical_route_dow_delay_rate",

    # V2: reliable context
    "reliable_origin_hour_delay_rate",
    "reliable_route_hour_delay_rate",
    "reliable_carrier_hour_delay_rate",

    "reliable_origin_dow_delay_rate",
    "reliable_route_dow_delay_rate",

    # V2: recent context
    "origin_hour_flights_last_7d",
    "origin_hour_delay_rate_last_7d",

    "route_hour_flights_last_7d",
    "route_hour_delay_rate_last_7d",

    "carrier_hour_flights_last_7d",
    "carrier_hour_delay_rate_last_7d"
]

target_col = "is_delayed"

print("Number of features:", len(feature_cols))

# COMMAND ----------

train_sample = train.sample(
    withReplacement=False,
    fraction=300000 / train.count(),
    seed=42
)

validation_sample = validation.sample(
    withReplacement=False,
    fraction=236000 / validation.count(),
    seed=42
)

test_sample = test.sample(
    withReplacement=False,
    fraction=175000 / test.count(),
    seed=42
)

print("Train:", train_sample.count())
print("Validation:", validation_sample.count())
print("Test:", test_sample.count())

# COMMAND ----------

train_pd = train_sample.select(
    feature_cols + [target_col]
).toPandas()

validation_pd = validation_sample.select(
    feature_cols + [target_col]
).toPandas()

test_pd = test_sample.select(
    feature_cols + [target_col]
).toPandas()

print("Train:", train_pd.shape)
print("Validation:", validation_pd.shape)
print("Test:", test_pd.shape)

# COMMAND ----------

missing = (
    train_pd[feature_cols]
    .isna()
    .sum()
    .sort_values(ascending=False)
)

print(missing[missing > 0])

# COMMAND ----------

import numpy as np

print(
    "Total NaNs:",
    train_pd[feature_cols].isna().sum().sum()
)

print(
    "Total Inf:",
    np.isinf(
        train_pd[feature_cols].select_dtypes(include=np.number)
    ).sum().sum()
)

# COMMAND ----------

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

X_train_raw = train_pd[feature_cols].copy()
X_val_raw = validation_pd[feature_cols].copy()
X_test_raw = test_pd[feature_cols].copy()

y_train = train_pd[target_col]
y_val = validation_pd[target_col]
y_test = test_pd[target_col]

# -------------------------
# 1. IMPUTATION
# -------------------------

imputer = SimpleImputer(strategy="median")

X_train_imp = imputer.fit_transform(X_train_raw)
X_val_imp = imputer.transform(X_val_raw)
X_test_imp = imputer.transform(X_test_raw)

# -------------------------
# 2. SCALING
# -------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train_imp)
X_val = scaler.transform(X_val_imp)
X_test = scaler.transform(X_test_imp)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("NaNs:", np.isnan(X_train).sum())

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

lr_v2 = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="lbfgs",
    random_state=42
)

lr_v2.fit(X_train, y_train)

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

lr_v2_prob = lr_v2.predict_proba(X_val)[:, 1]
lr_v2_pred = (lr_v2_prob >= 0.5).astype(int)

print("LOGISTIC REGRESSION V2")
print("---------------------")
print("Accuracy :", accuracy_score(y_val, lr_v2_pred))
print("Precision:", precision_score(y_val, lr_v2_pred))
print("Recall   :", recall_score(y_val, lr_v2_pred))
print("F1       :", f1_score(y_val, lr_v2_pred))
print("ROC-AUC  :", roc_auc_score(y_val, lr_v2_prob))
print("PR-AUC   :", average_precision_score(y_val, lr_v2_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, lr_v2_pred))

# COMMAND ----------

for threshold in np.arange(0.10, 0.51, 0.05):
    pred = (lr_v2_prob >= threshold).astype(int)

    print(
        f"Threshold {threshold:.2f} | "
        f"Precision: {precision_score(y_val, pred):.3f} | "
        f"Recall: {recall_score(y_val, pred):.3f} | "
        f"F1: {f1_score(y_val, pred):.3f}"
    )

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

rf_v2 = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_v2.fit(X_train, y_train)

# COMMAND ----------

rf_v2_prob = rf_v2.predict_proba(X_val)[:, 1]
rf_v2_pred = (rf_v2_prob >= 0.5).astype(int)

print("RANDOM FOREST V2")
print("----------------")
print("Accuracy :", accuracy_score(y_val, rf_v2_pred))
print("Precision:", precision_score(y_val, rf_v2_pred))
print("Recall   :", recall_score(y_val, rf_v2_pred))
print("F1       :", f1_score(y_val, rf_v2_pred))
print("ROC-AUC  :", roc_auc_score(y_val, rf_v2_prob))
print("PR-AUC   :", average_precision_score(y_val, rf_v2_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, rf_v2_pred))

# COMMAND ----------

for threshold in np.arange(0.10, 0.51, 0.05):
    pred = (rf_v2_prob >= threshold).astype(int)

    print(
        f"Threshold {threshold:.2f} | "
        f"Precision: {precision_score(y_val, pred):.3f} | "
        f"Recall: {recall_score(y_val, pred):.3f} | "
        f"F1: {f1_score(y_val, pred):.3f}"
    )

# COMMAND ----------

# MAGIC %pip install xgboost-cpu==3.3.0

# COMMAND ----------

import xgboost as xgb

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("Negative:", negative)
print("Positive:", positive)
print("Scale positive weight:", scale_pos_weight)

# COMMAND ----------

xgb_v2 = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_v2.fit(X_train, y_train)

# COMMAND ----------

xgb_v2_prob = xgb_v2.predict_proba(X_val)[:, 1]
xgb_v2_pred = (xgb_v2_prob >= 0.5).astype(int)

print("XGBOOST V2 — VALIDATION")
print("-----------------------")
print("Accuracy :", accuracy_score(y_val, xgb_v2_pred))
print("Precision:", precision_score(y_val, xgb_v2_pred))
print("Recall   :", recall_score(y_val, xgb_v2_pred))
print("F1       :", f1_score(y_val, xgb_v2_pred))
print("ROC-AUC  :", roc_auc_score(y_val, xgb_v2_prob))
print("PR-AUC   :", average_precision_score(y_val, xgb_v2_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, xgb_v2_pred))

# COMMAND ----------

final_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

final_rf.fit(
    X_train,
    y_train
)

# COMMAND ----------

prob = final_rf.predict_proba(X_val)[:,1]

for threshold in [0.25,0.3,0.35,0.4,0.45,0.5]:

    pred = (prob >= threshold).astype(int)

    print(
        threshold,
        "Precision:",
        precision_score(y_val,pred),
        "Recall:",
        recall_score(y_val,pred),
        "F1:",
        f1_score(y_val,pred)
    )

# COMMAND ----------

# ============================================================
# FINAL RANDOM FOREST TRAINING + TEST EVALUATION
# ============================================================

import numpy as np
import pandas as pd
import pickle
import json
import time

from sklearn.ensemble import RandomForestClassifier

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


# ============================================================
# 1. TRAIN FINAL MODEL ON FULL TRAIN DATA
# ============================================================

print("Training final Random Forest...")


start = time.time()


final_rf = RandomForestClassifier(

    n_estimators=300,

    max_depth=12,

    min_samples_leaf=20,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1

)


final_rf.fit(
    X_train,
    y_train
)


print(
    "Training time:",
    round((time.time()-start)/60,2),
    "minutes"
)



# ============================================================
# 2. VALIDATION CHECK (THRESHOLD 0.40)
# ============================================================


VAL_THRESHOLD = 0.40


val_prob = final_rf.predict_proba(
    X_val
)[:,1]


val_pred = (
    val_prob >= VAL_THRESHOLD
).astype(int)


print("\nVALIDATION")
print("----------------")

print(
    "Precision:",
    precision_score(y_val,val_pred)
)

print(
    "Recall:",
    recall_score(y_val,val_pred)
)

print(
    "F1:",
    f1_score(y_val,val_pred)
)

print(
    "ROC-AUC:",
    roc_auc_score(y_val,val_prob)
)

print(
    "PR-AUC:",
    average_precision_score(y_val,val_prob)
)

print(
    "\nConfusion Matrix"
)

print(
    confusion_matrix(
        y_val,
        val_pred
    )
)



# ============================================================
# 3. FINAL TEST EVALUATION
# ============================================================


test_prob = final_rf.predict_proba(
    X_test
)[:,1]


test_pred = (
    test_prob >= VAL_THRESHOLD
).astype(int)



print("\n\nFINAL TEST RESULTS")
print("------------------")


metrics = {

    "accuracy":
    accuracy_score(
        y_test,
        test_pred
    ),

    "precision":
    precision_score(
        y_test,
        test_pred
    ),

    "recall":
    recall_score(
        y_test,
        test_pred
    ),

    "f1":
    f1_score(
        y_test,
        test_pred
    ),

    "roc_auc":
    roc_auc_score(
        y_test,
        test_prob
    ),

    "pr_auc":
    average_precision_score(
        y_test,
        test_prob
    ),

    "threshold":
    VAL_THRESHOLD

}


print(metrics)


print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        test_pred
    )
)


print("\nClassification Report")

print(
    classification_report(
        y_test,
        test_pred
    )
)



# ============================================================
# 4. SAVE ARTIFACTS
# ============================================================


print("\nSaving artifacts...")


with open(
    "flight_delay_random_forest.pkl",
    "wb"
) as f:

    pickle.dump(
        final_rf,
        f
    )



with open(
    "flight_delay_feature_cols.json",
    "w"
) as f:

    json.dump(
        feature_cols,
        f,
        indent=4
    )



with open(
    "flight_delay_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


print("Saved successfully")