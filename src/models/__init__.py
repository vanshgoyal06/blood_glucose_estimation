from .catboost_model import CatBoostModel
from .lightgbm_model import LightGBMModel
from .xgboost_model import XGBoostModel
from .random_forest_model import RandomForestModel

MODEL_REGISTRY = {
    "catboost": CatBoostModel,
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
    "random_forest": RandomForestModel,
}
