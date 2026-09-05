import pandas as pd
from src.features.rolling_features import add_past_only_rolling_features

def test_rolling_is_past_only():
    df=pd.DataFrame({
        "Age":[1,1,1,1],
        "Diastolic Blood Pressure":[1,2,3,4],
        "Systolic Blood Pressure":[1,2,3,4],
        "Heart Rate":[10,20,30,40],
        "Body Temperature":[1,2,3,4],
        "SPO2":[90,91,92,93],
    })
    out=add_past_only_rolling_features(df,"Age",2)
    # row 2 uses rows 0 and 1, not row 2
    assert out.loc[2,"Heart Rate_roll2_mean"]==15
