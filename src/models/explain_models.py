from pathlib import Path
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt


DATA_FILE = Path("data/processed/fd001_train_features.csv")

RUL_MODEL_PATH = Path("models/xgb_rul_model.joblib")
FAILURE_MODEL_PATH = Path("models/xgb_failure_model.joblib")

FIGURE_DIR = Path("reports/figures")


def load_artifact(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    return joblib.load(path)


def create_shap_summary(
    model,
    X_sample: pd.DataFrame,
    output_path: Path,
    title: str
):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
        max_display=20
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP summary: {output_path}")


def create_feature_importance_table(
    model,
    feature_cols: list[str],
    output_path: Path
):
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    )

    importance_df.to_csv(output_path, index=False)

    print(f"Saved feature importance table: {output_path}")


def explain_models():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing feature file: {DATA_FILE}. "
            "Run src/features/build_features.py first."
        )

    df = pd.read_csv(DATA_FILE)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    rul_artifact = load_artifact(RUL_MODEL_PATH)
    failure_artifact = load_artifact(FAILURE_MODEL_PATH)

    rul_model = rul_artifact["model"]
    rul_features = rul_artifact["feature_cols"]

    failure_model = failure_artifact["model"]
    failure_features = failure_artifact["feature_cols"]

    # Sample for faster SHAP computation.
    # Fixed random state keeps the output reproducible.
    sample_df = df.sample(
        n=min(2000, len(df)),
        random_state=42
    )

    X_rul = sample_df[rul_features]
    X_failure = sample_df[failure_features]

    create_shap_summary(
        model=rul_model,
        X_sample=X_rul,
        output_path=FIGURE_DIR / "rul_shap_summary.png",
        title="RUL Model SHAP Summary"
    )

    create_shap_summary(
        model=failure_model,
        X_sample=X_failure,
        output_path=FIGURE_DIR / "failure_shap_summary.png",
        title="Failure Model SHAP Summary"
    )

    create_feature_importance_table(
        model=rul_model,
        feature_cols=rul_features,
        output_path=Path("reports/rul_feature_importance.csv")
    )

    create_feature_importance_table(
        model=failure_model,
        feature_cols=failure_features,
        output_path=Path("reports/failure_feature_importance.csv")
    )


if __name__ == "__main__":
    explain_models()