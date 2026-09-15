"""
LightGBM regressor wrapper. Mirrors notebook cell 13.
"""
import lightgbm as lgb

from config import LIGHTGBM_PARAMS, LIGHTGBM_NUM_ROUNDS, LIGHTGBM_EARLY_STOPPING_ROUNDS
from .base import BaseRegressor


class LightGBMModel(BaseRegressor):
    name = "LightGBM"

    def __init__(self, params: dict = None):
        self.params = params or LIGHTGBM_PARAMS
        self.booster = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val)
            valid_sets.append(val_data)
            valid_names.append("valid")

        callbacks = [lgb.early_stopping(stopping_rounds=LIGHTGBM_EARLY_STOPPING_ROUNDS, verbose=True)]

        self.booster = lgb.train(
            self.params,
            train_data,
            LIGHTGBM_NUM_ROUNDS,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    def predict(self, X):
        return self.booster.predict(X, num_iteration=self.booster.best_iteration)

    def feature_importance(self):
        return self.booster.feature_importance()
