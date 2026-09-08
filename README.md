# Aviation Risk Intelligence | ML & Analytics Platform

An end-to-end **flight delay prediction and analytics project** built using Databricks, PySpark, SQL, Scikit-learn, and Power BI. The project combines large-scale data processing, machine learning, model inference, and business analytics to identify flights with a higher risk of delay.

## Project Overview

Flight delays can vary significantly across airlines, routes, airports, departure times, and historical operating patterns. This project uses historical flight data to build a machine learning workflow that predicts flight delay risk and converts those predictions into actionable operational insights.

The pipeline processes **6.9M+ flight records**, engineers historical and temporal features, trains a Random Forest classification model, generates risk scores for unseen flights, and presents the results through an interactive Power BI dashboard.

## Tech Stack

* **Data Processing:** Databricks, PySpark
* **Analytics:** SQL, Python, Pandas
* **Machine Learning:** Scikit-learn, Random Forest
* **Visualization:** Power BI
* **Model Artifacts:** Pickle, JSON
* **Development:** Python, Jupyter/Databricks Notebooks

## Project Pipeline

```text
Flight Data
     ↓
Data Ingestion & Cleaning
     ↓
Feature Engineering
     ↓
Historical / Temporal Features
     ↓
Train / Validation / Test Split
     ↓
Model Training & Evaluation
     ↓
Flight Risk Scoring
     ↓
Business Analysis
     ↓
Power BI Dashboard
```

## Machine Learning

The model uses features covering:

* Flight timing and seasonality
* Route-level historical performance
* Origin airport performance
* Carrier performance
* Route, airport, and carrier delay rates
* Recent 7-day operating patterns
* Hour-of-day patterns
* Day-of-week patterns
* Flight distance

A **Random Forest classifier** was trained to predict whether a flight would be delayed.

### Model Performance

| Metric   |    Result |
| -------- | --------: |
| ROC-AUC  |  **0.65** |
| PR-AUC   |  **0.33** |
| Recall   | **63.5%** |
| F1 Score | **39.8%** |

The model uses a decision threshold of **0.40** to prioritize identifying potentially delayed flights.

## Risk Segmentation

Predicted flights are grouped into three operational risk categories:

| Risk Category | Risk Score  |
| ------------- | ----------- |
| Low           | < 0.30      |
| Medium        | 0.30 - 0.50 |
| High          | ≥ 0.50      |

On the December 2024 test set:

* **584,832 flights** were evaluated
* **21.15%** of flights experienced an actual delay
* High-risk flights represented **21.26%** of all flights
* High-risk flights contained **35.29%** of observed delays
* High-risk flights had a **35.1%** observed delay rate compared with **11.0%** for low-risk flights

## Business Analysis

The project analyzes flight delay patterns across multiple dimensions:

* **Airline performance**
* **Route-level delay rates**
* **Departure-hour patterns**
* **Daily delay trends**
* **Risk-category distribution**

These analyses help identify where delays are concentrated and how the model's risk scores relate to observed delay patterns.

## Power BI Dashboard

The Power BI dashboard provides an interactive view of:

* Overall flight and model KPIs
* Daily delay trends
* Flight risk distribution
* Delay rates by airline
* Top delay-prone routes
* Delay rates by departure hour


### Power BI Report

The complete Power BI report is available in the [`dashboard`](dashboard/) folder.

Download **`Flight_Delay_Analytics.pbix`** and open it using **Power BI Desktop** to explore the dashboard.

## Repository Structure

```text
flight-delay-prediction-analytics/
│
├── analysis/
│   ├── 01_data_ingestion
│   ├── 02_feature_engineering
│   ├── 03_model_training
│   ├── 04_model_inference
│   └── 05_business_analysis
│
├── saved_artifacts/
│   ├── flight_delay_random_forest.pkl
│   ├── flight_delay_imputer.pkl
│   ├── flight_delay_scaler.pkl
│   ├── feature_columns.json
│   └── threshold.json
│
├── dashboard/
│   └── Flight_Delay_Analytics.pbix
│
└── README.md
```

## Key Takeaways

* Built a complete ML workflow from data processing to business reporting.
* Processed **6.9M+ flight records** using Databricks and PySpark.
* Developed historical and temporal features for flight delay prediction.
* Evaluated model performance using ROC-AUC, PR-AUC, recall, and F1.
* Converted model outputs into operational risk categories.
* Used SQL and Power BI to translate model predictions into business insights.

## Author

**Garvit Audichya**
AI/ML Engineer | Data Science | Generative AI

