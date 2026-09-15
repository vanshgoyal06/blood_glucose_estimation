"""
Class-imbalance resampling strategies.
Mirrors notebook cell 6 (SMOTE) and cell 15 (ADASYN / RandomUnderSampler),
which were previously copy-pasted per cell.
"""
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

from config import RANDOM_STATE


def _resample(sampler, X: pd.DataFrame, y: pd.Series):
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled


def apply_smote(X: pd.DataFrame, y: pd.Series):
    return _resample(SMOTE(random_state=RANDOM_STATE), X, y)


def apply_adasyn(X: pd.DataFrame, y: pd.Series):
    return _resample(ADASYN(random_state=RANDOM_STATE), X, y)


def apply_undersampling(X: pd.DataFrame, y: pd.Series):
    return _resample(RandomUnderSampler(random_state=RANDOM_STATE), X, y)


STRATEGY_FUNCTIONS = {
    "smote": apply_smote,
    "adasyn": apply_adasyn,
    "undersample": apply_undersampling,
}


def apply_strategy(strategy: str, X: pd.DataFrame, y: pd.Series):
    """
    Dispatch to the right resampler by name. 'none' returns the data unchanged.
    Note: in the original notebook, resampling is driven off the Diabetic/NonDiabetic
    class label (y here should be that class column when calling this for training-set
    balancing), not the continuous BGL target.
    """
    if strategy == "none":
        return X, y
    if strategy not in STRATEGY_FUNCTIONS:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    return STRATEGY_FUNCTIONS[strategy](X, y)
