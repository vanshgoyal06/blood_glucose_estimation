import pandas as pd

def make_base_features(df):
    """Return the seven non-identifying wearable features used in the later baseline."""
    cols = [
        "Diastolic Blood Pressure",
        "Systolic Blood Pressure",
        "Heart Rate",
        "Body Temperature",
        "SPO2",
        "Sweating  (Y/N)",
        "Shivering (Y/N)",
    ]
    return df[cols].copy(), df["Blood Glucose Level(BGL)"].copy()

def add_engineered_features(df):
    out = df.copy()
    out["Pulse Pressure"] = out["Systolic Blood Pressure"] - out["Diastolic Blood Pressure"]
    out["MAP"] = out["Diastolic Blood Pressure"] + (
        out["Systolic Blood Pressure"] - out["Diastolic Blood Pressure"]
    ) / 3
    out["HR_Sweating"] = out["Heart Rate"] * out["Sweating  (Y/N)"]
    out["HR_Shivering"] = out["Heart Rate"] * out["Shivering (Y/N)"]
    out["BP_Ratio"] = out["Systolic Blood Pressure"] / out["Diastolic Blood Pressure"]
    return out
