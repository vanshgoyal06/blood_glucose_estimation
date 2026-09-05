from .catboost_models import make_regime_classifier

def train_stage1(X_train, y_train, X_val, y_val):
    model=make_regime_classifier()
    model.fit(X_train,(y_train>=200).astype(int),
              eval_set=(X_val,(y_val>=200).astype(int)),verbose=False)
    return model
