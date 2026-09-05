# Wearable-Based Blood Glucose Level (BGL) Estimation

## Project status

This repository packages the project developed during the conversation. It is **not** a claim of clinical validity.

The current primary model is the **single CatBoost regressor using 250-row past-only rolling mean/std features**. Its previously reported validation result on the legacy row-level split was:

- RMSE: **7.8812**
- MAE: **3.1898**
- R²: **0.9683**

These metrics are **within-dataset validation metrics**, not evidence of generalization to unseen participants.

A later participant-level 250-row evaluation was constructed during the packaging audit. Its results are stored separately and explicitly marked as a packaging-time audit.

## Objective

Estimate BGL from non-invasive wearable-style physiological measurements, while investigating whether historical physiological context improves prediction.

## Inputs

The primary rolling model uses past-only mean and standard deviation features for:

- Diastolic Blood Pressure
- Systolic Blood Pressure
- Heart Rate
- Body Temperature
- SPO2

The earlier point-in-time experiments also used Sweating and Shivering and investigated Age/diabetic status. The final rolling model deliberately uses only the 10 rolling statistics above.

Target: `Blood Glucose Level(BGL)`.

## Data

The supplied Excel dataset contains 16,969 raw rows before duplicate removal and 272 duplicate rows were removed in the working pipeline, leaving **16,697 rows**.

The dataset has 13 Age values/groups. In this project, `Age` has been used as a participant/group proxy because that was how the source data behaved in the conversation. **It is not independently established that Age is a unique participant identifier.** This is a methodological limitation.

The dataset source description says it combines true and synthetic records and that non-diabetic BGL has limited variation. Treat this as a dataset provenance limitation requiring independent verification.

## Pipeline

```text
Raw Excel
   ↓
Column cleanup + duplicate removal
   ↓
D/N encoding
   ↓
Past-only 250-row rolling mean/std
   ↓
History-based feature matrix
   ↓
Train / validation / test split
   ↓
CatBoost regression
   ↓
BGL prediction
   ↓
Evaluation + leakage audit
```

## Rolling-window definition

For each physiological variable:

1. Shift the series by one observation within the `Age` group.
2. Calculate a 250-observation rolling mean.
3. Calculate a 250-observation rolling standard deviation.
4. Require a complete 250-observation history.
5. Use the resulting features to predict the current row's BGL.

The one-row shift was explicitly audited: the model's rolling Heart Rate mean matched the manually constructed past-only mean, while it did not match the mean that included the current row.

**Important:** the implementation is past-only relative to the `Age`-group sequence, but it may cross the dataset's contiguous blocks. The semantic meaning of those blocks is not established.

## Models explored

- Random Forest
- XGBoost
- LightGBM
- CatBoost
- CatBoost with engineered point-in-time features
- CatBoost with log-transformed target
- 50/100/150/200/250-row physiological history
- High-BGL regime classifier
- Two-stage normal/high-BGL CatBoost
- Probability-weighted two-stage model
- Previous-BGL persistence baseline

The current primary model remains the single 250-row CatBoost because the analyzed two-stage alternatives had worse overall validation RMSE/R², although they dramatically improved high-BGL-regime error.

## Important findings

- Previous BGL exactly equaled current BGL for **84.06%** of usable observations.
- Previous-BGL persistence baseline: RMSE **5.1765**, MAE **0.9206**, R² **0.9857**.
- This means the dataset has very strong BGL persistence; high model scores must therefore be interpreted carefully.
- The 250-row single CatBoost achieved RMSE **7.8812**, MAE **3.1898**, R² **0.9683** on the legacy row-level validation split.
- The hard two-stage model achieved approximately RMSE **9.478**, MAE **1.895**, R² **0.9541**; high-BGL RMSE was approximately **0.31**.
- Probability weighting improved the two-stage overall result but remained behind the single model.
- A previously run 100-row participant-level validation gave approximately RMSE **40.02**, MAE **21.06**, R² **0.168**, demonstrating that row-level validation and unseen-participant evaluation answer very different questions.
- A packaging-time 250-row participant-level evaluation is included in `results/metrics/leakage_audit_results.json`.

## Reproducibility

Install:

```bash
pip install -r requirements.txt
```

Run the primary legacy reproduction:

```bash
python train.py
```

Run the structural/leakage audit and participant-level evaluation:

```bash
python audit.py
```

The scripts expect:

```text
data/raw/dataset.xlsx
```

## Existing vs new participant

### Existing participant with sufficient history

If enough historical wearable observations are available for the participant/group, a 250-observation history can be constructed and passed to the model.

### New participant

The current project does **not** establish that a completely new participant can receive the same performance immediately.

Operational concept:

```text
New participant
    ↓
Collect observations
    ↓
Accumulate sufficient history
    ↓
Construct 250-observation history
    ↓
Generate rolling features
    ↓
Predict BGL
```

The minimum history for the current rolling feature implementation is **250 previous observations for a complete window**.

## Clinical / deployment caveat

This is an ML research prototype, not a medical device. The supplied dataset is small, has limited participant coverage, contains synthetic/low-variation records according to its source description, and has unresolved questions around block/session semantics and participant identity. Clinical validation, external validation, prospective data collection, calibration, uncertainty estimation, safety analysis, and regulatory work would be required before real-world medical use.

## Repository structure

See `docs/project_handoff.md`, `docs/methodology.md`, and `docs/leakage_audit.md`.
