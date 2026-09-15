"""
CatBoost regressor wrapper. Mirrors notebook cells 12, 17, 18.
"""
from catboost import CatBoostRegressor

from config import CATBOOST_PARAMS
from .base import BaseRegressor


class CatBoostModel(BaseRegressor):
    name = "CatBoost"

    def __init__(self, params: dict = None):
        self.model = CatBoostRegressor(**(params or CATBOOST_PARAMS))

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def feature_importance(self):
        return self.model.get_feature_importance()
