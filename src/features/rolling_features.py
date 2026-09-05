import pandas as pd

def add_past_only_rolling_features(df, participant_col="Age", window=250):
    """
    Reproduce the project's current 250-row construction:
    shift by one row within participant/group, then calculate rolling mean/std.

    IMPORTANT: this respects the row order within each participant/group but does
    not know whether separated blocks represent distinct sessions. That limitation
    is documented in docs/leakage_audit.md.
    """
    out = df.copy()
    phys_features = [
        "Diastolic Blood Pressure",
        "Systolic Blood Pressure",
        "Heart Rate",
        "Body Temperature",
        "SPO2",
    ]
    for feature in phys_features:
        shifted = out.groupby(participant_col)[feature].shift(1)
        out[f"{feature}_roll{window}_mean"] = (
            shifted.groupby(out[participant_col])
            .rolling(window=window, min_periods=window)
            .mean()
            .reset_index(level=0, drop=True)
        )
        out[f"{feature}_roll{window}_std"] = (
            shifted.groupby(out[participant_col])
            .rolling(window=window, min_periods=window)
            .std()
            .reset_index(level=0, drop=True)
        )
    return out

def rolling_feature_names(window=250):
    features = [
        "Diastolic Blood Pressure",
        "Systolic Blood Pressure",
        "Heart Rate",
        "Body Temperature",
        "SPO2",
    ]
    return [f"{f}_roll{window}_{s}" for f in features for s in ("mean","std")]
