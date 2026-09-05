import pandas as pd
from src.data.preprocessing import add_engineered_features

def test_engineered_features():
    df=pd.DataFrame({"Systolic Blood Pressure":[120],"Diastolic Blood Pressure":[80],
                     "Heart Rate":[70],"Sweating  (Y/N)":[1],"Shivering (Y/N)":[0]})
    out=add_engineered_features(df)
    assert out.loc[0,"Pulse Pressure"]==40
    assert round(out.loc[0,"MAP"],6)==93.333333
    assert out.loc[0,"HR_Sweating"]==70
    assert out.loc[0,"HR_Shivering"]==0
