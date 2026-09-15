"""
Common interface every model wrapper implements, so experiments/run_experiment.py
can treat CatBoost / LightGBM / XGBoost / RandomForest identically.
"""
from abc import ABC, abstractmethod


class BaseRegressor(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the underlying model. X_val/y_val used for early stopping where supported."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    @abstractmethod
    def feature_importance(self):
        raise NotImplementedError
