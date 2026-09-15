"""
Run a single (model, sampling strategy) experiment end-to-end -- i.e. reproduce
one row of the Results table in the PPT.

Usage:
    python -m experiments.run_experiment --model catboost --sampling adasyn
    python -m experiments.run_experiment --model xgboost --sampling smote --tune
"""
import argparse

from config import TARGET_COLUMN, FEATURES
from src.data_loader import load_clean_dataset
from src.feature_engineering import add_rolling_features
from src.resampling import apply_strategy
from src.preprocessing import split_data, scale_features
from src.evaluate import regression_metrics, ResultsTable
from src.models import MODEL_REGISTRY
from src.utils import set_seed


def run(model_name: str, sampling: str, tune: bool = False) -> dict:
    set_seed()

    df = load_clean_dataset()
    df = add_rolling_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, FEATURES, TARGET_COLUMN)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)

    if sampling != "none":
        X_train_scaled, y_train = apply_strategy(sampling, X_train_scaled, y_train)

    model_cls = MODEL_REGISTRY[model_name]
    kwargs = {"tune": tune} if model_name == "xgboost" else {}
    model = model_cls(**kwargs)
    model.fit(X_train_scaled, y_train, X_val_scaled, y_val)

    val_metrics = regression_metrics(y_val, model.predict(X_val_scaled))
    test_metrics = regression_metrics(y_test, model.predict(X_test_scaled))

    print(f"[{model.name} | {sampling}] Validation:", val_metrics)
    print(f"[{model.name} | {sampling}] Test:", test_metrics)

    return {"model": model, "val_metrics": val_metrics, "test_metrics": test_metrics}


def main():
    parser = argparse.ArgumentParser(description="Run one BGL model experiment.")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    parser.add_argument("--sampling", choices=["none", "smote", "adasyn", "undersample"], default="none")
    parser.add_argument("--tune", action="store_true", help="Grid-search hyperparameters (XGBoost only)")
    args = parser.parse_args()

    result = run(args.model, args.sampling, args.tune)

    table = ResultsTable()
    table.add(result["model"].name, args.sampling, result["val_metrics"], result["test_metrics"])
    print(table.to_dataframe())


if __name__ == "__main__":
    main()
