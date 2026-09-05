# Project Handoff

## Current state

The project has progressed from a point-in-time regression baseline to a history-aware CatBoost model and a substantial reliability audit.

### Current primary model

Single CatBoost + 250-row past-only physiological rolling mean/std features.

Previously reported legacy validation:
- RMSE 7.8812
- MAE 3.1898
- R² 0.9683

### Not the final clinical claim

These are not clinically validated or independently generalizable metrics.

## Completed

- Dataset loading and cleaning
- Duplicate removal
- D/N encoding
- Point-in-time model comparison
- Feature-engineering experiments
- 50/100/150/200/250 rolling-history experiments
- Past-only rolling-window audit
- BGL persistence baseline
- Block-structure analysis
- Legacy split overlap analysis
- Participant-level split construction
- High-BGL regime experiment
- Two-stage and probability-weighted experiments
- Modular packaging of the primary methodology
- Manager presentation

## In validation / audit

- Meaning of the 701 derived contiguous blocks
- Whether Age is truly a participant identifier or only an age field
- Whether histories should be blocked at session boundaries
- Generalization to unseen participants
- External validity

## Packaging-time audit result

A 250-row participant-level train/validation/test evaluation was run while packaging:
- Train groups: 14, 15, 19, 46
- Validation groups: 9, 55
- Test group: 76
- Validation RMSE: 41.3116
- Validation MAE: 19.9567
- Validation R²: 0.1210
- Test RMSE: 33.6528
- Test MAE: 17.3586
- Test R²: 0.4139

These are **new packaging-time audit results**, not results previously reported in the conversation. They inherit the unresolved block/session issue.

## Reproduction

```bash
pip install -r requirements.txt
python train.py
python audit.py
```

`train.py` reproduces the legacy primary validation result exactly from the packaged dataset and also computes a newly generated legacy-split test result.

## Important files

- `train.py` — primary legacy model reproduction
- `audit.py` — structural/leakage audit + participant-level audit
- `src/features/rolling_features.py` — 250-row feature construction
- `src/models/catboost_models.py` — CatBoost models
- `docs/leakage_audit.md` — reliability assessment
- `docs/methodology.md` — technical method
- `presentation/project_presentation.pptx` — manager presentation
- `notebooks/` — original working notebooks/snapshots
- `results/metrics/` — machine-readable metrics

## Recommended next steps

1. Establish the semantics of the 701 blocks.
2. Decide whether block-aware windows are scientifically appropriate.
3. Re-run the primary experiment with the corrected history construction.
4. Use repeated participant-level validation.
5. Freeze the methodology.
6. Evaluate the untouched test set.
7. Only after this, optimize deployment/model size.

## Reviewer questions

- Is Age actually a participant identifier?
- What does a contiguous block represent?
- Why is 250 observations the right history length?
- Does the model add information beyond previous BGL?
- How does performance change on unseen participants?
- How much performance comes from long constant-BGL runs?
- Are synthetic/non-diabetic records dominating the metrics?
- Can the result generalize outside this dataset?
