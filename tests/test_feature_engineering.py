"""
Basic sanity tests for the rolling-window feature engineering and classification
logic that was previously untested in the notebook.

Run with: python -m pytest tests/
"""
import pandas as pd

from src.feature_engineering import add_rolling_features, classify_diabetes


def test_add_rolling_features_creates_expected_columns():
    df = pd.DataFrame(
        {
            "Diastolic_Blood_Pressure": range(10),
            "Systolic_Blood_Pressure": range(10),
            "Heart_Rate": range(10),
            "Body_Temperature": [36.5] * 10,
            "SPO2": [98] * 10,
        }
    )
    result = add_rolling_features(df, window=3)

    assert "Diastolic_Blood_Pressure_rolling_mean" in result.columns
    assert "Diastolic_Blood_Pressure_rolling_std" in result.columns
    assert result.isna().sum().sum() == 0


def test_classify_diabetes_threshold():
    bgl = pd.Series([100, 126, 200, 50])
    labels = classify_diabetes(bgl, threshold=126.0)

    assert list(labels) == ["Non-Diabetic", "Diabetic", "Diabetic", "Non-Diabetic"]
