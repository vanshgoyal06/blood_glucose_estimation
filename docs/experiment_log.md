# Experiment Log

This log records the important experiments actually reported during the project conversation.

| Experiment | Result | Interpretation |
|---|---|---|
| Initial CatBoost, point-in-time | RMSE ≈ 40, MAE ≈ 22, R² ≈ 0.225 | Weak cross-row/general row-level baseline before history features |
| CatBoost without Age | RMSE 38.4503, MAE 21.3977, R² 0.2247 | Removing Age slightly improved RMSE/MAE |
| CatBoost without Age + diabetic status | RMSE 38.4456, MAE 21.2890, R² 0.2249 | Tiny RMSE/R² change; lower MAE |
| 50-row history | RMSE 16.3465, MAE 8.8191, R² 0.8620 | Longitudinal context strongly improves in-dataset validation |
| 100-row history | RMSE 12.5176, MAE 6.0134, R² 0.9199 | Further improvement |
| 150-row history | RMSE 10.3802, MAE 4.4542, R² 0.9454 | Further improvement |
| 200-row history | RMSE 8.5374, MAE 3.6258, R² 0.9635 | Further improvement |
| 250-row history | RMSE 7.8812, MAE 3.1898, R² 0.9683 | Current primary historical result |
| Previous-BGL persistence | RMSE 5.1765, MAE 0.9206, R² 0.9857 | BGL persistence is exceptionally strong |
| 100-row participant-level validation | RMSE ≈ 40.02, MAE ≈ 21.06, R² ≈ 0.168 | Unseen-group generalization is much harder |
| Two-stage CatBoost | RMSE ≈ 9.478, MAE ≈ 1.895, R² ≈ 0.9541 | Better MAE and high-BGL subset, worse overall RMSE/R² |
| Probability-weighted two-stage | RMSE ≈ 9.251, MAE ≈ 2.355, R² ≈ 0.9563 | Better than hard switch but still behind single model |

## High-BGL finding

The high-BGL regime is narrow (roughly 242–250 in the analyzed data). The specialized high-BGL regressor achieved RMSE ≈ 0.31 on the legacy validation high-BGL subset.

This is a useful diagnostic, not evidence that the overall system is clinically accurate.

## Packaging-time audit

The package reran the participant-level 250-row design:
- Train groups: 14, 15, 19, 46
- Validation groups: 9, 55
- Test group: 76
- Validation RMSE 41.3116, MAE 19.9567, R² 0.1210
- Test RMSE 33.6528, MAE 17.3586, R² 0.4139

These were not historical results from the conversation; they were computed during packaging to complete the requested sanity check.
