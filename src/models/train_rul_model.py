from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


DATA_FILE = Path("data/processed/fd001_train_features.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xgb_rul_model.joblib"

TARGET = "RUL_capped"

DROP_COLUMNS = [
    "unit_number",
    "time_in_cycles",
    "RUL",
    "RUL_capped",
    "failure_within_30_cycles",
]


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing feature file: {DATA_FILE}. "
            "Run src/features/build_features.py first."
        )

    return pd.read_csv(DATA_FILE)


def train_test_split_by_engine(df: pd.DataFrame):
    """
    Splits by engine unit, not by rows.
    This prevents leakage from the same engine appearing in both train and test.
    """
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


def get_features_and_target(df: pd.DataFrame):
    feature_cols = [
        col for col in df.columns
        if col not in DROP_COLUMNS
    ]

    X = df[feature_cols]
    y = df[TARGET]

    return X, y, feature_cols


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("RUL regression performance:")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2:   {r2:.3f}")

    return preds


def train_model():
    df = load_data()

    train_df, test_df = train_test_split_by_engine(df)

    X_train, y_train, feature_cols = get_features_and_target(train_df)
    X_test, y_test, _ = get_features_and_target(test_df)

    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = evaluate_model(model, X_test, y_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "target": TARGET,
        },
        MODEL_PATH
    )

    results = test_df[["unit_number", "time_in_cycles", "RUL", "RUL_capped"]].copy()
    results["predicted_RUL"] = preds

    output_path = Path("reports/rul_predictions_sample.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Prediction sample saved to: {output_path}")


if __name__ == "__main__":
    train_model()