from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

from src.data.loading import load_dataset, clean_dataset
from src.features.rolling_features import add_past_only_rolling_features, rolling_feature_names
from src.models.catboost_models import train_single_regressor

df_raw=load_dataset(ROOT/"data/raw/dataset.xlsx")
df, duplicates=clean_dataset(df_raw)
df_roll=add_past_only_rolling_features(df,"Age",250)
roll_cols=rolling_feature_names(250)
clean=df_roll.dropna(subset=roll_cols).copy()

# Dataset structural audit.
tmp=df.copy()
tmp["Block_ID"]=((
    (tmp["Age"]!=tmp["Age"].shift()) |
    (tmp["Diabetic/NonDiabetic (D/N)"]!=tmp["Diabetic/NonDiabetic (D/N)"].shift())
).cumsum()).astype(int)

# Participant split used in the latest project audit.
train_p=[14,15,19,46]
val_p=[9,55]
test_p=[76]

def subset(participants):
    idx=clean.index[clean["Age"].isin(participants)]
    return clean.loc[idx,roll_cols], clean.loc[idx,"Blood Glucose Level(BGL)"]

Xtr,ytr=subset(train_p)
Xva,yva=subset(val_p)
Xte,yte=subset(test_p)

model=train_single_regressor(Xtr,ytr,Xva,yva)
pv=model.predict(Xva)
pt=model.predict(Xte)

def mets(y,p):
    return {
        "RMSE":float(np.sqrt(mean_squared_error(y,p))),
        "MAE":float(mean_absolute_error(y,p)),
        "R2":float(r2_score(y,p))
    }

# Past-only check on a representative row with complete history.
rep_idx=int(clean.index[300])
age=df.loc[rep_idx,"Age"]
previous=df[(df["Age"]==age)&(df.index<rep_idx)].tail(250)
current=df[(df["Age"]==age)&(df.index<=rep_idx)].tail(250)
heart_model=float(clean.loc[rep_idx,"Heart Rate_roll250_mean"])
heart_past=float(previous["Heart Rate"].mean())
heart_inc=float(current["Heart Rate"].mean())

# Count training rows appearing in validation history for the legacy row split.
# This is an audit of the old split, not of the clean participant split.
from sklearn.model_selection import train_test_split
base_cols=[
    "Diastolic Blood Pressure","Systolic Blood Pressure","Heart Rate",
    "Body Temperature","SPO2","Sweating  (Y/N)","Shivering (Y/N)"
]
X=df[base_cols]; y=df["Blood Glucose Level(BGL)"]
Xtrb,Xtemp,ytrb,ytemp=train_test_split(X,y,test_size=.30,random_state=42,stratify=df.loc[X.index,"Age"])
Xv,Xt,yv,yt=train_test_split(Xtemp,ytemp,test_size=.50,random_state=42,
                             stratify=Xtemp.index.map(df["Diabetic/NonDiabetic (D/N)"]))
training_set=set(Xtrb.index)
rolling_validation_indices = clean.index.intersection(Xv.index)
overlap_counts=[]
for idx in rolling_validation_indices:
    a=df.loc[idx,"Age"]
    hist=df[(df["Age"]==a)&(df.index<idx)].tail(250)
    overlap_counts.append(sum(i in training_set for i in hist.index))

report={
    "dataset":{"raw_rows":len(df_raw),"duplicates_removed":duplicates,"clean_rows":len(df),
               "unique_age_groups":int(df["Age"].nunique()),"blocks":int(tmp["Block_ID"].nunique())},
    "past_only_representative_check":{
        "index":rep_idx,"age":int(age),"history_size":len(previous),
        "model_rolling_mean":heart_model,"manual_past_only_mean":heart_past,
        "manual_including_current_mean":heart_inc,
        "model_vs_past_only_abs_diff":abs(heart_model-heart_past),
        "model_vs_including_current_abs_diff":abs(heart_model-heart_inc)
    },
    "legacy_split_history_overlap":{
        "validation_rows_checked":len(rolling_validation_indices),
        "validation_rows_with_training_rows_in_250_history":int(sum(c>0 for c in overlap_counts)),
        "percentage":float(100*sum(c>0 for c in overlap_counts)/len(overlap_counts)),
        "mean_training_rows_in_history":float(np.mean(overlap_counts)),
        "max_training_rows_in_history":int(np.max(overlap_counts))
    },
    "clean_participant_split":{
        "train_participants":train_p,"validation_participants":val_p,"test_participants":test_p,
        "train_shape":list(Xtr.shape),"validation_shape":list(Xva.shape),"test_shape":list(Xte.shape),
        "validation":mets(yva,pv),"test":mets(yte,pt),
        "status":"packaging-time audit result; not previously reported in the conversation"
    },
    "block_structure":{
        "warning":"Block_ID is derived from contiguous Age + diabetic-status runs. Its semantic meaning as a session/device recording is not established.",
        "blocks_ge_250":int((tmp.groupby("Block_ID").size()>=250).sum()),
        "rows_in_blocks_ge_250":int(tmp.groupby("Block_ID").size().loc[lambda s:s>=250].sum())
    }
}
(ROOT/"results/metrics").mkdir(parents=True,exist_ok=True)
with open(ROOT/"results/metrics/leakage_audit_results.json","w",encoding="utf-8") as f:
    json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
