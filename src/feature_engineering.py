"""
Feature engineering: rolling-window vitals features + BGL classification label.
Mirrors notebook cells 4 and 8.
"""
import pandas as pd

from config import RAW_VITAL_COLUMNS, ROLLING_WINDOW, DIABETIC_THRESHOLD


def add_rolling_features(
    df: pd.DataFrame,
    columns=RAW_VITAL_COLUMNS,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """
    Add reverse rolling mean/std for each vital sign, then drop rows that
    don't have a full window (NaNs introduced at the tail).
    """
    df = df.copy()
    for col in columns:
        reversed_col = df[col].iloc[::-1]
        df[f"{col}_rolling_mean"] = reversed_col.rolling(window=window).mean().iloc[::-1]
        df[f"{col}_rolling_std"] = reversed_col.rolling(window=window).std().iloc[::-1]

    df = df.dropna()
    return df


def classify_diabetes(bgl_series: pd.Series, threshold: float = DIABETIC_THRESHOLD) -> pd.Series:
    """Label each BGL reading as Diabetic / Non-Diabetic using a clinical threshold."""
    return bgl_series.apply(lambda x: "Diabetic" if x >= threshold else "Non-Diabetic")
