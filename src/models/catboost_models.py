from catboost import CatBoostRegressor, CatBoostClassifier

def make_catboost_regressor(depth=7):
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=1000,
        learning_rate=0.03,
        depth=depth,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False,
    )

def make_regime_classifier():
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=1000,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=3,
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=False,
    )

def train_single_regressor(X_train, y_train, X_val=None, y_val=None):
    model=make_catboost_regressor()
    if X_val is None:
        model.fit(X_train,y_train,verbose=False)
    else:
        model.fit(X_train,y_train,eval_set=(X_val,y_val),verbose=False)
    return model

def train_two_stage(X_train, y_train, X_val, y_val, threshold=0.5):
    """Train the analyzed hard-switch and probability-weighted two-stage models."""
    high_train = y_train >= 200
    normal_train = ~high_train
    clf = make_regime_classifier()
    clf.fit(X_train, high_train.astype(int), eval_set=(X_val,(y_val>=200).astype(int)), verbose=False)
    normal = make_catboost_regressor()
    high = make_catboost_regressor()
    normal.fit(X_train.loc[normal_train], y_train.loc[normal_train], verbose=False)
    high.fit(X_train.loc[high_train], y_train.loc[high_train], verbose=False)
    p = clf.predict_proba(X_val)[:,1]
    normal_pred = normal.predict(X_val)
    high_pred = high.predict(X_val)
    hard = (p >= threshold)
    hard_pred = hard.astype(float)*high_pred + (~hard).astype(float)*normal_pred
    blend_pred = (1-p)*normal_pred + p*high_pred
    return clf, normal, high, p, normal_pred, high_pred, hard_pred, blend_pred
