# Gedis – Predictive Maintenance Demo

Gedis is a project focused on predictive maintenance using multivariate sensor time-series data.

## What the project predicts

- Remaining Useful Life (RUL)
- failure probability within the next 30 cycles
- unusual sensor behaviour
- maintenance risk score
- key features influencing model predictions

## Dataset

Dataset used: NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset.

This is a benchmark simulation dataset, not live factory data.

## Models

| Task | Model |
|---|---|
| RUL prediction | XGBoost Regressor |
| Failure prediction | XGBoost Classifier |
| Anomaly detection | Isolation Forest |
| Explainability | SHAP |
| Dashboard | Streamlit |

## Current results

### RUL regression

| Metric | Value |
|---|---:|
| MAE | 9.672 |
| RMSE | 14.233 |
| R² | 0.884 |

### Failure classification

| Metric | Value |
|---|---:|
| Accuracy | 0.957 |
| Precision | 0.816 |
| Recall | 0.929 |
| F1-score | 0.869 |
| ROC-AUC | 0.994 |

## Feature engineering

Main engineered features:

- rolling sensor mean
- rolling sensor standard deviation
- sensor delta from previous cycle
- cycles from start
- capped RUL target
- removed low-variance sensors

I also checked for data leakage. Features based on the true maximum engine life were removed because they would not be available at prediction time.

## Maintenance risk score

The maintenance risk score combines:

```text
risk_score = 0.50 * failure_probability
           + 0.35 * RUL_risk
           + 0.15 * anomaly_risk

Risk levels:

| Score | Risk level |
|---:|---|
| 0–25 | Low |
| 25–50 | Medium |
| 50–75 | High |
| 75–100 | Critical |

This score is a project heuristic, not a validated maintenance rule.

## Dashboard

The Streamlit dashboard shows:

- selected engine unit
- predicted RUL
- failure probability
- anomaly score
- maintenance risk score
- sensor trends
- fleet-level risk overview
- SHAP explainability plots
