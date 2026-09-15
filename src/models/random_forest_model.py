"""
Random Forest regressor wrapper. Mirrors notebook cell 16.
"""
from sklearn.ensemble import RandomForestRegressor

from config import RANDOM_FOREST_PARAMS
from .base import BaseRegressor


class RandomForestModel(BaseRegressor):
    name = "Random Forest"

    def __init__(self, params: dict = None):
        self.model = RandomForestRegressor(**(params or RANDOM_FOREST_PARAMS))

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def feature_importance(self):
        return self.model.feature_importances_
