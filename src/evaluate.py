"""
Consistent regression metrics + results table builder.
Replaces the manual print(...) calls repeated across every model cell.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from config import RESULTS_DIR


def regression_metrics(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "r2": r2}


class ResultsTable:
    """Accumulates one row per (model, sampling strategy) run, like the PPT results slide."""

    def __init__(self):
        self.rows = []

    def add(self, model_name: str, sampling: str, val_metrics: dict, test_metrics: dict):
        self.rows.append(
            {
                "Model": model_name,
                "Sampling / Resampling Strategy": sampling,
                "Validation RMSE": val_metrics.get("rmse"),
                "Validation R2": val_metrics.get("r2"),
                "Test RMSE": test_metrics.get("rmse"),
                "Test R2": test_metrics.get("r2"),
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save(self, filename="results_table.csv"):
        df = self.to_dataframe()
        df.to_csv(RESULTS_DIR / filename, index=False)
        print(f"Saved results table to {RESULTS_DIR / filename}")
        return df
