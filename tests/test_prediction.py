import numpy as np
from src.evaluation.metrics import regression_metrics

def test_metrics_keys():
    m=regression_metrics(np.array([1,2,3]),np.array([1,2,4]))
    assert set(m)=={"RMSE","MAE","R2"}
