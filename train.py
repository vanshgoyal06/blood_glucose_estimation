from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.loading import load_dataset, clean_dataset
from src.data.splitting import random_row_split
from src.features.rolling_features import add_past_only_rolling_features, rolling_feature_names
from src.models.catboost_models import train_single_regressor
from src.evaluation.metrics import regression_metrics

DATA = ROOT / "data" / "raw" / "dataset.xlsx"
OUT = ROOT / "results"
(OUT / "metrics").mkdir(parents=True, exist_ok=True)
(OUT / "predictions").mkdir(parents=True, exist_ok=True)

df_raw = load_dataset(DATA)
df, duplicates = clean_dataset(df_raw)

feature_columns = [
    "Diastolic Blood Pressure", "Systolic Blood Pressure", "Heart Rate",
    "Body Temperature", "SPO2", "Sweating  (Y/N)", "Shivering (Y/N)"
]
target = "Blood Glucose Level(BGL)"

# Legacy/current primary 250-row construction and original row split.
df_roll = add_past_only_rolling_features(df, participant_col="Age", window=250)
roll_cols = rolling_feature_names(250)
df_roll = df_roll.dropna(subset=roll_cols).copy()

# Build the exact original row split on the seven wearable features, then intersect indices.
X_base = df[feature_columns]
y_base = df[target]
from sklearn.model_selection import train_test_split
X_train_base, X_temp_base, y_train_base, y_temp_base = train_test_split(
    X_base, y_base, test_size=0.30, random_state=42, stratify=df.loc[X_base.index, "Age"]
)
X_val_base, X_test_base, y_val_base, y_test_base = train_test_split(
    X_temp_base, y_temp_base, test_size=0.50, random_state=42,
    stratify=X_temp_base.index.map(df["Diabetic/NonDiabetic (D/N)"])
)

train_idx = df_roll.index.intersection(X_train_base.index)
val_idx = df_roll.index.intersection(X_val_base.index)
test_idx = df_roll.index.intersection(X_test_base.index)

X_train = df_roll.loc[train_idx, roll_cols]
y_train = df_roll.loc[train_idx, target]
X_val = df_roll.loc[val_idx, roll_cols]
y_val = df_roll.loc[val_idx, target]
X_test = df_roll.loc[test_idx, roll_cols]
y_test = df_roll.loc[test_idx, target]

model = train_single_regressor(X_train, y_train, X_val, y_val)
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

val_metrics = regression_metrics(y_val, val_pred)
test_metrics = regression_metrics(y_test, test_pred)

pd.DataFrame({"Actual_BGL": y_val.values, "Predicted_BGL": val_pred}).to_csv(
    OUT / "predictions" / "legacy_random_split_validation_predictions.csv", index=False
)
pd.DataFrame({"Actual_BGL": y_test.values, "Predicted_BGL": test_pred}).to_csv(
    OUT / "predictions" / "legacy_random_split_test_predictions.csv", index=False
)

result = {
    "experiment": "250-row single CatBoost, legacy row-level split",
    "duplicates_removed": duplicates,
    "rows_after_cleaning": len(df),
    "train_samples": len(X_train),
    "validation_samples": len(X_val),
    "test_samples": len(X_test),
    "validation": val_metrics,
    "test": test_metrics,
    "note": "The validation metrics are the reported historical champion. The test metrics are newly computed by this packaging run and are not a previously reported project result."
}
with open(OUT / "metrics" / "legacy_random_split.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

model.save_model(str(ROOT / "models" / "catboost_250_legacy_random_split.cbm"))
print(json.dumps(result, indent=2))
