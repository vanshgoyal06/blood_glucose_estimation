# Data Dictionary

| Column | Meaning in project | Type | Primary use |
|---|---|---|---|
| Age | Age field used as participant/group proxy | numeric | Grouping / stratification; not a proven participant ID |
| Blood Glucose Level(BGL) | Blood glucose target | numeric | Target |
| Diastolic Blood Pressure | Diastolic BP | numeric | Current physiology + rolling history |
| Systolic Blood Pressure | Systolic BP | numeric | Current physiology + rolling history |
| Heart Rate | Heart rate | numeric | Current physiology + rolling history |
| Body Temperature | Body temperature | numeric | Current physiology + rolling history |
| SPO2 | Blood oxygen saturation | numeric | Current physiology + rolling history |
| Sweating  (Y/N) | Sweating indicator | binary | Earlier point-in-time experiments |
| Shivering (Y/N) | Shivering indicator | binary | Earlier point-in-time experiments |
| Diabetic/NonDiabetic (D/N) | Source-provided diabetic status | binary after encoding | Earlier models/stratification; not in primary rolling features |

## Engineered rolling fields

For each of DBP, SBP, HR, Body Temperature and SPO2:

- `<feature>_roll250_mean`
- `<feature>_roll250_std`

These summarize the 250 previous observations in the current Age/group sequence.

## Timestamp

No timestamp column is present in the supplied working dataframe. The original row index/order is therefore used as the sequence order. This is a limitation for real temporal interpretation.
