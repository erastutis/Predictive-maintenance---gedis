from pathlib import Path
import pandas as pd


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

INDEX_COLUMNS = ["unit_number", "time_in_cycles"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
COLUMN_NAMES = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS


def find_file(filename: str) -> Path:
    matches = list(RAW_DATA_DIR.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} inside {RAW_DATA_DIR}. "
            "Make sure the C-MAPSS dataset is extracted."
        )

    return matches[0]


def load_cmapss_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES
    )

    return df


def add_rul_labels(train_df: pd.DataFrame) -> pd.DataFrame:
    max_cycles = train_df.groupby("unit_number")["time_in_cycles"].max()

    train_df = train_df.copy()
    train_df["max_cycle"] = train_df["unit_number"].map(max_cycles)
    train_df["RUL"] = train_df["max_cycle"] - train_df["time_in_cycles"]
    train_df = train_df.drop(columns=["max_cycle"])

    return train_df


def add_failure_label(df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    df = df.copy()
    df[f"failure_within_{horizon}_cycles"] = (df["RUL"] <= horizon).astype(int)
    return df


def build_fd001_dataset() -> pd.DataFrame:
    train_path = find_file("train_FD001.txt")

    train_df = load_cmapss_file(train_path)
    train_df = add_rul_labels(train_df)
    train_df = add_failure_label(train_df, horizon=30)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DATA_DIR / "fd001_train_processed.csv"
    train_df.to_csv(output_path, index=False)

    print("FD001 training dataset created.")
    print(f"Source file: {train_path}")
    print(f"Output file: {output_path}")
    print(f"Shape: {train_df.shape}")
    print()
    print(train_df.head())

    return train_df


if __name__ == "__main__":
    build_fd001_dataset()