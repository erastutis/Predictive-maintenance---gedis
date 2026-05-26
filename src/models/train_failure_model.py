from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier


DATA_FILE = Path("data/processed/fd001_train_features.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xgb_failure_model.joblib"

TARGET = "failure_within_30_cycles"

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
    Splits data by engine unit to prevent leakage.
    Rows from the same engine must not appear in both train and test sets.
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
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print("Failure classification performance:")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"ROC-AUC:   {roc_auc:.3f}")
    print()
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print()
    print("Classification report:")
    print(classification_report(y_test, predictions))

    return probabilities, predictions


def train_model():
    df = load_data()

    train_df, test_df = train_test_split_by_engine(df)

    X_train, y_train, feature_cols = get_features_and_target(train_df)
    X_test, y_test, _ = get_features_and_target(test_df)

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count

    print(f"Training rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Positive class ratio train: {y_train.mean():.3f}")
    print(f"scale_pos_weight: {scale_pos_weight:.3f}")
    print()

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    probabilities, predictions = evaluate_model(model, X_test, y_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "target": TARGET,
            "threshold": 0.5,
        },
        MODEL_PATH
    )

    results = test_df[
        ["unit_number", "time_in_cycles", "RUL", "RUL_capped", TARGET]
    ].copy()

    results["failure_probability"] = probabilities
    results["predicted_failure"] = predictions

    output_path = Path("reports/failure_predictions_sample.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Prediction sample saved to: {output_path}")


if __name__ == "__main__":
    train_model()