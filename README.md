# Blood Glucose Level (BGL) Estimation from Wearable Data

Estimating Blood Glucose Level (BGL) from non-invasive wearable/smartwatch vitals
(Heart Rate, SPO2, Blood Pressure, Body Temperature) using classical ML regressors.

Refactored from `project_v1.ipynb` into a proper, reusable Python package instead of
one linear notebook.

## Project structure

```
bgl_estimation_project/
├── README.md
├── requirements.txt
├── config.py                    # all constants: paths, feature lists, hyperparams
├── main.py                      # single entry point: runs the full pipeline
│
├── data/
│   └── dataset.xlsx             # raw data (place your copy here)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # read_dataset(), clean_dataset()
│   ├── feature_engineering.py   # add_rolling_features(), classify_diabetes()
│   ├── resampling.py            # apply_smote(), apply_adasyn(), apply_undersampling()
│   ├── preprocessing.py         # split_data(), scale_features()
│   ├── eda.py                   # plot_histograms(), plot_correlation_heatmap(), summarize_distribution()
│   ├── evaluate.py              # regression_metrics(), results table builder
│   ├── utils.py                 # logging setup, seed setting, save/load helpers
│   └── models/
│       ├── __init__.py
│       ├── base.py              # shared BaseRegressor interface
│       ├── catboost_model.py
│       ├── lightgbm_model.py
│       ├── xgboost_model.py
│       └── random_forest_model.py
│
├── experiments/
│   └── run_experiment.py        # runs one (model, sampling strategy) combo end-to-end
│
├── outputs/
│   ├── models/                  # saved .pkl / .cbm / .txt model artifacts
│   ├── figures/                 # saved EDA + evaluation plots
│   └── results/                 # results_table.csv, metrics.json (feeds your PPT table)
│
├── notebooks/
│   └── exploratory.ipynb        # keep ONLY for ad-hoc exploration, not the pipeline
│
└── tests/
    └── test_feature_engineering.py
```

## How the notebook maps onto this structure

| Notebook cell(s) | New location |
|---|---|
| Cell 0–2: load data, dtype conversion, dedup | `src/data_loader.py` |
| Cell 3, 5, 8: diabetic counts, BGL_Status classification | `src/feature_engineering.py::classify_diabetes` |
| Cell 4: rolling window (window=1500) features | `src/feature_engineering.py::add_rolling_features` |
| Cell 6: SMOTE resampling | `src/resampling.py::apply_smote` |
| Cell 9–10: histograms, correlation heatmap | `src/eda.py` |
| Cell 12, 17, 18: CatBoost (+ SMOTE/ADASYN variants) | `src/models/catboost_model.py` |
| Cell 13: LightGBM | `src/models/lightgbm_model.py` |
| Cell 14, 15: XGBoost (+ GridSearch, SMOTE/ADASYN/undersampling) | `src/models/xgboost_model.py` |
| Cell 16: Random Forest | `src/models/random_forest_model.py` |
| Metrics printed inline everywhere | `src/evaluate.py` (consistent RMSE/R² + results table) |

## Why this split

- **`config.py`** — every magic number in the notebook (window size 1500, test splits,
  feature lists, model hyperparameters) becomes one editable place instead of being
  retyped in 6 different cells.
- **`src/data_loader.py` vs `src/feature_engineering.py`** — loading/cleaning raw data
  is separated from creating derived features, so you can re-run feature engineering
  without re-reading the Excel file each time.
- **`src/resampling.py`** — SMOTE/ADASYN/undersampling logic was duplicated in 4 cells
  with copy-paste. It's now one tested function each, imported wherever needed.
- **`src/models/*`** — each model gets `train()` / `evaluate()` behind the same
  interface (`src/models/base.py`), so `experiments/run_experiment.py` can loop over
  `(model, sampling_strategy)` combinations instead of hand-writing a new cell per
  combination (this is exactly the 8-row table in your PPT's Results slide).
- **`experiments/run_experiment.py`** — reproduces one row of the results table.
  `main.py` loops over all combinations and writes `outputs/results/results_table.csv`.
- **`outputs/`** — nothing is printed-and-lost anymore; plots, models, and metrics are
  all saved to disk.
- **`tests/`** — rolling-window / resampling logic is easy to silently break; a couple
  of unit tests catch that early.

## Usage

```bash
pip install -r requirements.txt

# put dataset.xlsx in data/

# run everything (EDA + all model/sampling combinations)
python main.py

# or run just one experiment, e.g. CatBoost + ADASYN
python -m experiments.run_experiment --model catboost --sampling adasyn
```

## Roadmap (from the PPT's "Future Work")

- Swap in a richer time-series dataset (e.g. PhysioCGM) — see `config.py::DATASET_SOURCE`.
- Add `src/models/deep/` for CNN-LSTM / TabNet / Transformer models once sequential
  (not just tabular) data is available.
