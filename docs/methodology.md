# Methodology

## 1. Problem framing

The project investigates whether wearable-style physiological measurements can be used to estimate BGL, with special emphasis on historical context. The target is the current row's `Blood Glucose Level(BGL)`.

The project eventually aims toward smartwatch/fitness-band deployment, but the present work is a research/engineering prototype and does not establish medical-device readiness.

## 2. Dataset preparation

The working notebook:
- loads the Excel workbook;
- strips whitespace from column names;
- counts and removes duplicate rows;
- resets the index;
- maps diabetic status `D` to 1 and `N` to 0.

The cleaned dataset contains 16,697 rows after removal of 272 duplicates.

No scaling was required for the tree-based models used in the primary experiments.

## 3. Point-in-time baseline

The initial experiments used:
- Age
- Diastolic Blood Pressure
- Systolic Blood Pressure
- Heart Rate
- Body Temperature
- SPO2
- Sweating
- Shivering
- Diabetic/NonDiabetic status

Models compared included Random Forest, XGBoost, LightGBM and CatBoost.

Feature engineering experiments also tested Pulse Pressure, MAP, HR×Sweating, HR×Shivering and SBP/DBP ratio.

## 4. Historical feature engineering

The key idea was to replace an isolated physiological observation with summary statistics over a participant/group's recent history.

For each of five physiological variables:
- Diastolic Blood Pressure
- Systolic Blood Pressure
- Heart Rate
- Body Temperature
- SPO2

the implementation performs:

```text
shift(1)
    ↓
group by Age
    ↓
rolling(window=250, min_periods=250)
    ├── mean
    └── standard deviation
```

The one-row shift ensures the current observation is not included in its own rolling features.

## 5. Why the 250-row model was selected

The same CatBoost configuration was evaluated with multiple history lengths. The reported validation trend was:

| History | RMSE | MAE | R² |
|---|---:|---:|---:|
| 50 | 16.3465 | 8.8191 | 0.8620 |
| 100 | 12.5176 | 6.0134 | 0.9199 |
| 150 | 10.3802 | 4.4542 | 0.9454 |
| 200 | 8.5374 | 3.6258 | 0.9635 |
| 250 | 7.8812 | 3.1898 | 0.9683 |

These comparisons used the project's legacy row-level split and are not proof of independent generalization.

## 6. Primary model

The current primary model is:

**CatBoostRegressor**
- loss: RMSE
- iterations: 1000
- learning rate: 0.03
- depth: 7
- L2 regularization: 3
- random seed: 42

Input: ten 250-row rolling mean/std features.

Output: estimated current BGL.

## 7. Baselines and regime analysis

The project also evaluated a previous-BGL persistence baseline. It achieved:

- RMSE 5.1765
- MAE 0.9206
- R² 0.9857
- exact previous=current rate 84.06%

This is a crucial diagnostic: much of the apparent predictability in this dataset comes from BGL persistence.

The data also contains a large high-BGL regime around 242–250, while there is a gap in the observed BGL distribution between 130 and 241 in the analyzed block structure.

A regime-aware experiment trained:
1. a high/normal BGL classifier (`>=200` vs `<200`);
2. a normal-BGL regressor;
3. a high-BGL regressor.

Hard switching and probability-weighted blending were both tested. The single 250-row model remained the overall champion in the reported comparison, although the specialized high-BGL regressor had extremely low error on the high-BGL subset.

## 8. Validation designs

### Legacy row-level validation

The original experiment split rows into 70/15/15 using `train_test_split`, with Age used for stratification and diabetic status used for the second split's stratification.

This design permits observations from the same Age/group to occur in train, validation and test. It is therefore not a participant-independent generalization test.

### Participant-level design

A later design separated groups:
- Train: 14, 15, 19, 46
- Validation: 9, 55
- Test: 76

The packaging-time audit produced:
- train: 4,700
- validation: 7,572
- test: 2,212

The resulting metrics are stored separately and are explicitly marked as packaging-time audit results, not historical conversation results.

## 9. Rolling-window audit

A representative audit row showed:
- model Heart Rate rolling mean = 102.084
- manual past-only mean = 102.084
- difference = 0
- mean including current row = 102.076

Therefore the one-row shift/past-only construction is behaving as intended for that check.

## 10. The major unresolved issue: blocks

The raw row order contains 701 contiguous blocks when a new block is defined by a change in Age or diabetic status.

Many blocks are short, while several are very long. In the supplied analysis:
- 83.59% of blocks had exactly one BGL value;
- the largest block contained 1,151 rows;
- only three blocks had at least 250 rows.

The 250-row implementation groups by Age, not Block_ID. Therefore, if the blocks represent separate sessions/recordings, the rolling history can cross those boundaries. The project has **not established what the blocks semantically represent**.

This is a WARNING, not a claim that leakage definitely occurs.

## 11. Target leakage and current features

The primary rolling features are physiological measurements only. The current primary model does not use current BGL or previous BGL as an input feature.

However, the dataset itself has strong temporal BGL persistence. That makes the task easier even without explicit BGL-history features.

## 12. Two different operational tasks

### Existing participant with sufficient history

A participant/group with enough prior wearable observations can generate the 250-row rolling feature vector.

### New participant

The current implementation cannot claim equal performance immediately for a completely new participant. The operational sequence is:

```text
New participant
    ↓
Collect observations
    ↓
Accumulate 250 previous observations
    ↓
Construct rolling features
    ↓
Run trained model
    ↓
Estimate BGL
```

The 250 observations are a model requirement in the current implementation, not a clinical recommendation.

## 13. Interpretation

The most defensible interpretation is not simply "the model predicts BGL with R²=0.9683."

Instead:

> The model performs very strongly on the legacy in-dataset validation setup, but the dataset's strong BGL persistence, repeated blocks, limited participant coverage, and unresolved block/session semantics mean that independent generalization remains the key validation question.

## 14. Future work

Future work should include:
- confirm the meaning of the 701 blocks;
- construct block/session-aware rolling windows if justified;
- perform repeated participant-level cross-validation;
- evaluate an untouched test set only after the methodology is frozen;
- assess performance separately on diabetic and non-diabetic/source-defined subsets;
- evaluate high-BGL event sensitivity;
- obtain external/prospective data;
- quantify uncertainty and calibration;
- develop a production inference interface;
- only then investigate deployment and clinical/regulatory requirements.
