"""
Load and clean the raw wearable-vitals dataset.
Mirrors notebook cells 0-2 (load, dtype conversion, dedup).
"""
import pandas as pd

from config import DATASET_PATH, NUMERIC_COLUMNS


def read_dataset(path=DATASET_PATH) -> pd.DataFrame:
    """Read the raw Excel dataset and normalize column names."""
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Type-convert numeric columns and drop duplicate patient rows."""
    df = df.copy()
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")

    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Dropped {before - after} duplicate rows ({before} -> {after}).")

    df = df.dropna(subset=[NUMERIC_COLUMNS[0]])  # drop rows with missing BGL target
    return df


def load_clean_dataset(path=DATASET_PATH) -> pd.DataFrame:
    """Convenience wrapper: read + clean in one call."""
    return clean_dataset(read_dataset(path))
