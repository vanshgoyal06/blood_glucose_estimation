# Leakage and Reliability Audit

## Executive status

| Check | Status | Finding |
|---|---|---|
| Current-row inclusion in rolling features | PASS for audited construction | One-row shift was verified against a manual past-only calculation. |
| Future information inside the same rolling calculation | PASS for audited construction | The rolling calculation is shifted by one observation. |
| Participant overlap in the clean participant split | PASS | Train/validation/test groups are disjoint in the chosen design. |
| Legacy row-level participant separation | FAIL for participant-independent evaluation | Same Age/group can appear in multiple splits. |
| Legacy validation history containing training rows | WARNING / CONFIRMED | 2,188 of 2,189 validation rows had training rows in their 250-row history. |
| Rolling-window overlap | WARNING | Adjacent samples share most of their history. |
| Block/session boundary handling | NOT VERIFIED | 250-row features group by Age and can cross derived contiguous blocks. |
| Target leakage from current BGL | PASS for primary feature list | Current BGL is not used in the rolling feature columns. |
| Previous-BGL leakage | NOT APPLICABLE to primary model | Previous BGL is used only as a diagnostic baseline, not as a primary model input. |
| Scaling/preprocessing leakage | LOW RISK / NOT APPLICABLE | Tree model does not require scaling; categorical encoding is deterministic. |
| Resampling leakage | NOT APPLICABLE to primary 250-row model | No resampling is used in the primary model. |
| Two-stage validation leakage | WARNING | Stage 2 models are trained only on training rows, but the overall experiment uses the legacy split and therefore inherits its evaluation limitations. |
| External validation | NOT VERIFIED | No independent external dataset has been evaluated. |

## 1. Rolling-window construction

The current implementation uses:

```python
shifted = df.groupby("Age")[feature].shift(1)
rolling = shifted.groupby(df["Age"]).rolling(window=250, min_periods=250)
```

The explicit shift means the current row is excluded.

A representative audit matched the model rolling mean to a manual mean of the 250 previous rows exactly.

**Conclusion:** the rolling operation itself is past-only.

## 2. Participant leakage in the legacy split

The original row-level split is not participant independent. The validation set can contain rows from the same Age/group as training.

In the conversation, the audit reported 2,188 of 2,189 validation rows with training rows in their 250-row history (99.95%), with a mean of about 166 training rows in the history.

When the packaged dataset and split logic were reconstructed during final packaging, the same check produced 2,189 of 2,189 (100.0%), with a mean of 174.53 training rows in the history. This small discrepancy indicates that the exact notebook state used for the earlier audit was not fully reconstructable from the surviving artifacts. Both results point to the same conclusion: the legacy validation histories are heavily populated by training observations.

This does not mean the current row's target leaked into its own feature. It means the validation sample's historical context can contain observations that were also part of model training.

## 3. Rolling-window overlap

A sliding window creates highly correlated samples by design.

For example:

```text
Prediction t
[ t-250 ... t-1 ]

Prediction t+1
[ t-249 ... t ]
```

These windows share 249 observations.

This is acceptable for a time-series/personalized inference problem, but it means row-level sample counts overstate the amount of independent information.

## 4. Block/session boundary risk

The dataset has 701 contiguous blocks when a block is defined by a change in Age or diabetic status.

The project has not established whether these blocks are:
- sessions,
- recording periods,
- device runs,
- data-generation batches,
- or simply dataset organization.

Because the current rolling implementation groups by Age, it can stitch separated blocks belonging to the same Age/group into one history.

This is the most important unresolved structural issue.

## 5. BGL persistence

The previous-BGL baseline achieved:

- RMSE 5.1765
- MAE 0.9206
- R² 0.9857
- 84.06% exact previous-to-current matches

Therefore, excellent regression metrics cannot automatically be interpreted as strong evidence of a physiological wearable-to-BGL relationship.

## 6. Generalization

The earlier 100-row participant-level experiment reported:

- RMSE ≈ 40.02
- MAE ≈ 21.06
- R² ≈ 0.168

This is dramatically weaker than the legacy row-level result of approximately:

- RMSE 7.88
- R² 0.968

The gap demonstrates that evaluation design matters enormously.

A packaging-time 250-row participant-level audit is included in `results/metrics/leakage_audit_results.json`.

## 7. Reliability conclusion

The reported 7.8812 / 3.1898 / 0.9683 result is **real as a reproduced validation result**, but it should be described as a **legacy in-dataset validation result**.

It should not be presented as independently validated generalization performance.

## 8. Recommended correction

Before calling the model final:

1. Establish what a block represents.
2. Freeze a session/block-aware ordering rule if appropriate.
3. Construct histories without crossing inappropriate boundaries.
4. Use participant-level evaluation.
5. Use repeated group/time-aware validation where data volume permits.
6. Keep the final test set untouched until the methodology is frozen.
