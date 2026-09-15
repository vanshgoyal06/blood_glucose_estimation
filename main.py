"""
Full pipeline entry point.

Runs EDA, then every (model, sampling strategy) combination reported in the PPT's
Results slide, and writes outputs/results/results_table.csv.

Usage:
    python main.py
"""
from config import TARGET_COLUMN, FEATURES
from src.data_loader import load_clean_dataset
from src.feature_engineering import add_rolling_features, classify_diabetes
from src.eda import summarize_distribution, plot_histograms, plot_correlation_heatmap
from src.resampling import apply_strategy
from src.preprocessing import split_data, scale_features
from src.evaluate import regression_metrics, ResultsTable
from src.models import MODEL_REGISTRY
from src.utils import set_seed, save_model

# (model_name, sampling_strategy) combinations matching the PPT results table
EXPERIMENTS = [
    ("catboost", "none"),
    ("catboost", "smote"),
    ("catboost", "adasyn"),
    ("lightgbm", "none"),
    ("xgboost", "smote"),
    ("xgboost", "adasyn"),
    ("xgboost", "undersample"),
    ("random_forest", "smote"),
]


def run_eda(df):
    df = df.copy()
    df["BGL_Status"] = classify_diabetes(df[TARGET_COLUMN])
    summarize_distribution(df, "INITIAL RAW DATASET")
    plot_histograms(df)
    plot_correlation_heatmap(df)


def main():
    set_seed()

    print("Loading and cleaning dataset...")
    df = load_clean_dataset()

    print("Running EDA...")
    run_eda(df)

    print(f"Building rolling-window features (window={1500})...")
    df_features = add_rolling_features(df)

    results = ResultsTable()

    for model_name, sampling in EXPERIMENTS:
        print(f"\n=== Running {model_name} | sampling={sampling} ===")

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df_features, FEATURES, TARGET_COLUMN
        )
        X_train_scaled, X_val_scaled, X_test_scaled, _ = scale_features(X_train, X_val, X_test)

        if sampling != "none":
            X_train_scaled, y_train = apply_strategy(sampling, X_train_scaled, y_train)

        model = MODEL_REGISTRY[model_name]()
        model.fit(X_train_scaled, y_train, X_val_scaled, y_val)

        val_metrics = regression_metrics(y_val, model.predict(X_val_scaled))
        test_metrics = regression_metrics(y_test, model.predict(X_test_scaled))

        results.add(model.name, sampling, val_metrics, test_metrics)
        save_model(model, f"{model_name}_{sampling}")

    final_table = results.save()
    print("\nFinal results table:")
    print(final_table)


if __name__ == "__main__":
    main()
