from pathlib import Path
import joblib
import pandas as pd
import numpy as np


DATA_FILE = Path("data/processed/fd001_train_features.csv")

RUL_MODEL_PATH = Path("models/xgb_rul_model.joblib")
FAILURE_MODEL_PATH = Path("models/xgb_failure_model.joblib")
ANOMALY_MODEL_PATH = Path("models/isolation_forest_anomaly_model.joblib")

OUTPUT_FILE = Path("reports/maintenance_risk_scores.csv")


def load_artifact(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")

    return joblib.load(path)


def min_max_scale(series: pd.Series) -> pd.Series:
    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(0.0, index=series.index)

    return (series - min_value) / (max_value - min_value)


def assign_risk_level(score: float) -> str:
    if score >= 75:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 25:
        return "Medium"
    return "Low"


def create_risk_scores():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing feature file: {DATA_FILE}. "
            "Run src/features/build_features.py first."
        )

    df = pd.read_csv(DATA_FILE)

    rul_artifact = load_artifact(RUL_MODEL_PATH)
    failure_artifact = load_artifact(FAILURE_MODEL_PATH)
    anomaly_artifact = load_artifact(ANOMALY_MODEL_PATH)

    rul_model = rul_artifact["model"]
    rul_features = rul_artifact["feature_cols"]

    failure_model = failure_artifact["model"]
    failure_features = failure_artifact["feature_cols"]

    anomaly_model = anomaly_artifact["model"]
    anomaly_scaler = anomaly_artifact["scaler"]
    anomaly_features = anomaly_artifact["feature_cols"]

    X_rul = df[rul_features]
    X_failure = df[failure_features]
    X_anomaly = df[anomaly_features]

    predicted_rul = rul_model.predict(X_rul)
    predicted_rul = np.clip(predicted_rul, 0, None)
    failure_probability = failure_model.predict_proba(X_failure)[:, 1]

    X_anomaly_scaled = anomaly_scaler.transform(X_anomaly)
    anomaly_score_raw = anomaly_model.decision_function(X_anomaly_scaled)
    anomaly_score = -anomaly_score_raw

    # Convert model outputs into comparable risk components.
    # Lower RUL = higher risk.
    rul_risk = 1 - min_max_scale(pd.Series(predicted_rul))

    # Higher anomaly score = higher risk.
    anomaly_risk = min_max_scale(pd.Series(anomaly_score))

    failure_risk = pd.Series(failure_probability)

    # Weighted maintenance risk score.
    # Failure probability is strongest because it is directly trained on near-failure target.
    # RUL contributes operational urgency.
    # Anomaly score contributes unsupervised sensor abnormality.
    maintenance_risk_score = (
        0.50 * failure_risk +
        0.35 * rul_risk +
        0.15 * anomaly_risk
    ) * 100

    results = df[
        [
            "unit_number",
            "time_in_cycles",
            "RUL",
            "RUL_capped",
            "failure_within_30_cycles",
        ]
    ].copy()

    results["predicted_RUL"] = predicted_rul
    results["failure_probability"] = failure_probability
    results["anomaly_score"] = anomaly_score
    results["maintenance_risk_score"] = maintenance_risk_score
    results["risk_level"] = results["maintenance_risk_score"].apply(assign_risk_level)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print("Maintenance risk scoring complete.")
    print(f"Output file: {OUTPUT_FILE}")
    print()
    print("Risk level distribution:")
    print(results["risk_level"].value_counts())
    print()
    print("Highest risk machine states:")
    print(
        results.sort_values("maintenance_risk_score", ascending=False)
        .head(15)
        [
            [
                "unit_number",
                "time_in_cycles",
                "RUL",
                "predicted_RUL",
                "failure_probability",
                "anomaly_score",
                "maintenance_risk_score",
                "risk_level",
            ]
        ]
    )


if __name__ == "__main__":
    create_risk_scores()