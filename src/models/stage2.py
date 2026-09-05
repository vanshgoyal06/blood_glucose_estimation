from .catboost_models import make_catboost_regressor

def train_stage2_regressors(X_train, y_train):
    high=y_train>=200
    normal=~high
    normal_model=make_catboost_regressor()
    high_model=make_catboost_regressor()
    normal_model.fit(X_train.loc[normal],y_train.loc[normal],verbose=False)
    high_model.fit(X_train.loc[high],y_train.loc[high],verbose=False)
    return normal_model, high_model
