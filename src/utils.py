"""
Shared small helpers: seeding, model persistence.
"""
import random
import numpy as np
import joblib

from config import RANDOM_STATE, MODELS_DIR


def set_seed(seed: int = RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)


def save_model(model, name: str):
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"Saved model to {path}")
    return path


def load_model(name: str):
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)
