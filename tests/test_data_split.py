import pandas as pd
from src.data.splitting import participant_split

def test_participant_sets_disjoint():
    df=pd.DataFrame({"Age":[1,1,2,2,3,3],"x":[1,2,3,4,5,6],"Blood Glucose Level(BGL)":[10,11,12,13,14,15]})
    tr,va,te,ytr,yva,yte=participant_split(df,["x"],[1],[2],[3])
    assert len(tr)==2 and len(va)==2 and len(te)==2
    assert set(tr.index).isdisjoint(va.index)
    assert set(va.index).isdisjoint(te.index)
