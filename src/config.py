from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.xlsx"
RANDOM_SEED = 42
HISTORY_WINDOW = 250
TARGET = "Blood Glucose Level(BGL)"
PARTICIPANT_COLUMN = "Age"  # In this dataset Age is used as the participant/group proxy; semantics are not independently established.

PHYS_FEATURES = [
    "Diastolic Blood Pressure",
    "Systolic Blood Pressure",
    "Heart Rate",
    "Body Temperature",
    "SPO2",
]

STATIC_WEARABLE_FEATURES = [
    "Diastolic Blood Pressure",
    "Systolic Blood Pressure",
    "Heart Rate",
    "Body Temperature",
    "SPO2",
    "Sweating  (Y/N)",
    "Shivering (Y/N)",
]

ROLLING_FEATURES = [
    f"{feature}_roll250_{stat}"
    for feature in PHYS_FEATURES
    for stat in ("mean", "std")
]
