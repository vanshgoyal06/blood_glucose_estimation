"""
Train/validation/test splitting and feature scaling.
Mirrors the "Step 2 / Step 3" blocks repeated in every model cell (12-18).
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import FEATURES, TARGET_COLUMN, TEST_SIZE, VAL_SIZE_OF_TEMP, RANDOM_STATE


def split_data(df: pd.DataFrame, features=FEATURES, target=TARGET_COLUMN):
    """80% train / 10% validation / 10% test split."""
    X = df[features]
    y = df[target]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE_OF_TEMP, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """Fit StandardScaler on train only, apply to val/test (no leakage)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
