from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


DATA_FILE = Path("data/processed/fd001_train_features.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "isolation_forest_anomaly_model.joblib"


NON_FEATURE_COLUMNS = [
    "unit_number",
    "time_in_cycles",
    "RUL",
    "RUL_capped",
    "failure_within_30_cycles",
    "cycles_from_start",
]


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing feature file: {DATA_FILE}. "
            "Run src/features/build_features.py first."
        )

    return pd.read_csv(DATA_FILE)


def split_by_engine(df: pd.DataFrame):
    groups = df["unit_number"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    return train_df, test_df


def get_anomaly_features(df: pd.DataFrame):
    """
    Uses sensor-derived features only.

    We intentionally remove lifecycle/time features because anomaly detection
    should identify abnormal sensor behaviour, not just early or late lifecycle
    positions.
    """
    feature_cols = [
        col for col in df.columns
        if col not in NON_FEATURE_COLUMNS
        and (
            col.startswith("sensor_")
            or "_roll_mean_" in col
            or "_roll_std_" in col
            or col.endswith("_delta")
        )
    ]

    X = df[feature_cols]

    return X, feature_cols


def train_anomaly_model():
    df = load_data()

    train_df, test_df = split_by_engine(df)

    # Train mostly on healthy operating states.
    # Avoid first few cycles because rolling features are still stabilizing.
    healthy_train_df = train_df[
        (train_df["RUL"] > 80) &
        (train_df["time_in_cycles"] > 20)
    ].copy()

    X_train, feature_cols = get_anomaly_features(healthy_train_df)
    X_test, _ = get_anomaly_features(test_df)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = IsolationForest(
        n_estimators=300,
        contamination=0.08,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled)

    anomaly_label = model.predict(X_test_scaled)
    anomaly_score_raw = model.decision_function(X_test_scaled)

    # Lower raw score = more anomalous.
    # Convert to intuitive score: higher = more anomalous.
    anomaly_score = -anomaly_score_raw

    results = test_df[
        ["unit_number", "time_in_cycles", "RUL", "RUL_capped", "failure_within_30_cycles"]
    ].copy()

    results["anomaly_label"] = anomaly_label
    results["is_anomaly"] = (anomaly_label == -1).astype(int)
    results["anomaly_score"] = anomaly_score

    # Main summary
    anomaly_rate = results["is_anomaly"].mean()
    anomaly_rate_near_failure = results.loc[results["RUL"] <= 30, "is_anomaly"].mean()
    anomaly_rate_healthy = results.loc[results["RUL"] > 80, "is_anomaly"].mean()

    # Additional summary without warm-up cycles
    stable_results = results[results["time_in_cycles"] > 20].copy()
    stable_anomaly_rate = stable_results["is_anomaly"].mean()
    stable_anomaly_rate_near_failure = stable_results.loc[
        stable_results["RUL"] <= 30, "is_anomaly"
    ].mean()
    stable_anomaly_rate_healthy = stable_results.loc[
        stable_results["RUL"] > 80, "is_anomaly"
    ].mean()

    print("Anomaly detection summary:")
    print(f"Test rows: {len(results)}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Overall anomaly rate: {anomaly_rate:.3f}")
    print(f"Anomaly rate when RUL <= 30: {anomaly_rate_near_failure:.3f}")
    print(f"Anomaly rate when RUL > 80:  {anomaly_rate_healthy:.3f}")
    print()
    print("Stable-cycle summary, excluding first 20 cycles:")
    print(f"Stable rows: {len(stable_results)}")
    print(f"Stable overall anomaly rate: {stable_anomaly_rate:.3f}")
    print(f"Stable anomaly rate when RUL <= 30: {stable_anomaly_rate_near_failure:.3f}")
    print(f"Stable anomaly rate when RUL > 80:  {stable_anomaly_rate_healthy:.3f}")
    print()
    print("Top anomalous samples:")
    print(
        stable_results.sort_values("anomaly_score", ascending=False)
        .head(10)
        [["unit_number", "time_in_cycles", "RUL", "is_anomaly", "anomaly_score"]]
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "contamination": 0.08,
        },
        MODEL_PATH
    )

    output_path = Path("reports/anomaly_predictions_sample.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print()
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Prediction sample saved to: {output_path}")


if __name__ == "__main__":
    train_anomaly_model()