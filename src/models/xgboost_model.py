"""
XGBoost regressor wrapper, with optional GridSearchCV tuning.
Mirrors notebook cells 14-15.
"""
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

from config import XGBOOST_PARAM_GRID
from .base import BaseRegressor


class XGBoostModel(BaseRegressor):
    name = "XGBoost"

    def __init__(self, tune: bool = False, param_grid: dict = None):
        self.tune = tune
        self.param_grid = param_grid or XGBOOST_PARAM_GRID
        self.model = xgb.XGBRegressor(objective="reg:squarederror")
        self.best_params_ = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.tune:
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=self.param_grid,
                scoring="neg_mean_squared_error",
                cv=3,
                verbose=1,
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)
            self.best_params_ = grid_search.best_params_
            print("Best Hyperparameters:", self.best_params_)
            print("Best CV RMSE:", np.sqrt(-grid_search.best_score_))
            self.model = xgb.XGBRegressor(**self.best_params_)

        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def feature_importance(self):
        return self.model.feature_importances_
