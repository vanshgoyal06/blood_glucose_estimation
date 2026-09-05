import pandas as pd
from pathlib import Path

def load_dataset(path):
    """Load the supplied Excel dataset without reordering rows."""
    path = Path(path)
    df = pd.read_excel(path)
    return df

def clean_dataset(df):
    """Apply the cleaning steps used in the working notebook."""
    out = df.copy()
    out.columns = out.columns.str.strip()
    duplicate_count = int(out.duplicated().sum())
    out = out.drop_duplicates().reset_index(drop=True)
    out["Diabetic/NonDiabetic (D/N)"] = (
        out["Diabetic/NonDiabetic (D/N)"].map({"D": 1, "N": 0})
    )
    return out, duplicate_count
