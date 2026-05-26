from pathlib import Path
import pandas as pd
import numpy as np


PROCESSED_DATA_DIR = Path("data/processed")

INPUT_FILE = PROCESSED_DATA_DIR / "fd001_train_processed.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "fd001_train_features.csv"

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]


def remove_constant_sensors(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Removes sensors with almost no variation.
    These features are not useful for ML models.
    """
    removed = []

    for col in SENSOR_COLUMNS:
        if df[col].nunique() <= 2:
            removed.append(col)

    df = df.drop(columns=removed)

    return df, removed


def add_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds lifecycle-based features.
    """
    df = df.copy()

    max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")

    df["cycle_ratio"] = df["time_in_cycles"] / max_cycles
    df["cycles_from_start"] = df["time_in_cycles"]
    df["estimated_life"] = max_cycles

    return df


def add_rolling_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    windows: list[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    Adds rolling mean and rolling standard deviation per engine unit.
    """
    df = df.copy()
    df = df.sort_values(["unit_number", "time_in_cycles"])

    for window in windows:
        for col in sensor_cols:
            df[f"{col}_roll_mean_{window}"] = (
                df.groupby("unit_number")[col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

            df[f"{col}_roll_std_{window}"] = (
                df.groupby("unit_number")[col]
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
            )

    return df


def add_delta_features(df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
    """
    Adds first-order sensor differences.
    This captures short-term sensor movement.
    """
    df = df.copy()
    df = df.sort_values(["unit_number", "time_in_cycles"])

    for col in sensor_cols:
        df[f"{col}_delta"] = (
            df.groupby("unit_number")[col]
            .diff()
            .fillna(0)
        )

    return df


def add_capped_rul(df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    """
    Caps RUL target.
    In predictive maintenance, very high RUL values are often capped because
    the early healthy stage is hard to distinguish precisely.
    """
    df = df.copy()
    df["RUL_capped"] = np.minimum(df["RUL"], cap)

    return df


def build_features() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}. "
            "Run src/data/make_dataset.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    df, removed_sensors = remove_constant_sensors(df)

    active_sensor_cols = [
        col for col in SENSOR_COLUMNS
        if col in df.columns
    ]

    df = add_cycle_features(df)
    df = add_rolling_features(df, active_sensor_cols, windows=[5, 10, 20])
    df = add_delta_features(df, active_sensor_cols)
    df = add_capped_rul(df, cap=125)

    df = df.fillna(0)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("Feature dataset created.")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Original sensor count: {len(SENSOR_COLUMNS)}")
    print(f"Removed low-variance sensors: {removed_sensors}")
    print(f"Final shape: {df.shape}")

    return df


if __name__ == "__main__":
    build_features()