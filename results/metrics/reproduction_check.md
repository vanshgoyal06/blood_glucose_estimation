# Reproduction Check

Packaging verification date: 2026-09-05.

`python train.py` reproduced the historical primary validation metrics exactly from the packaged dataset:

- RMSE: 7.8812336027397345
- MAE: 3.189838397816726
- R²: 0.9682877470214366

The script also generated a test result for the legacy split:

- RMSE: 8.093505563798855
- MAE: 3.306314439326643
- R²: 0.9657833815854433

The test result above was not previously reported in the conversation; it is included only as a packaging-time reproducibility check.

`pytest -q`: **4 passed**.
