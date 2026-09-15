"""
Central configuration for the BGL estimation project.
Every constant that used to be retyped across notebook cells lives here.
"""
from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DATASET_PATH = DATA_DIR / "dataset.xlsx"

OUTPUT_DIR = ROOT_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

for d in (MODELS_DIR, FIGURES_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Data / feature engineering ----------------------------------------
RAW_VITAL_COLUMNS = [
    "Diastolic_Blood_Pressure",
    "Systolic_Blood_Pressure",
    "Heart_Rate",
    "Body_Temperature",
    "SPO2",
]

NUMERIC_COLUMNS = [
    "Blood_Glucose_Level(BGL)",
    "Diastolic_Blood_Pressure",
    "Systolic_Blood_Pressure",
    "Heart_Rate",
    "Body_Temperature",
    "SPO2",
]

TARGET_COLUMN = "Blood_Glucose_Level(BGL)"
CLASS_COLUMN = "Diabetic/NonDiabetic_(D/N)"
DIABETIC_THRESHOLD = 126.0  # mg/dL, used for BGL_Status labeling in EDA

ROLLING_WINDOW = 1500

# Final feature set used by every model (rolling mean + std of each vital)
FEATURES = [f"{col}_rolling_mean" for col in RAW_VITAL_COLUMNS] + [
    f"{col}_rolling_std" for col in RAW_VITAL_COLUMNS
]

# --- Train / val / test split ------------------------------------------
TEST_SIZE = 0.2          # 20% held out, then split 50/50 -> 10% val / 10% test
VAL_SIZE_OF_TEMP = 0.5
RANDOM_STATE = 42

# --- Resampling ----------------------------------------------------------
SAMPLING_STRATEGIES = ["none", "smote", "adasyn", "undersample"]

# --- Model hyperparameters ----------------------------------------------
CATBOOST_PARAMS = dict(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function="RMSE",
    cat_features=[],
    verbose=200,
)

LIGHTGBM_PARAMS = dict(
    objective="regression",
    metric="rmse",
    boosting_type="gbdt",
    num_leaves=31,
    learning_rate=0.05,
    feature_fraction=0.9,
)
LIGHTGBM_NUM_ROUNDS = 1000
LIGHTGBM_EARLY_STOPPING_ROUNDS = 100

XGBOOST_PARAM_GRID = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7, 9],
    "n_estimators": [100, 200, 300],
    "colsample_bytree": [0.3, 0.5, 0.7],
    "alpha": [0, 1, 10],
}

RANDOM_FOREST_PARAMS = dict(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
