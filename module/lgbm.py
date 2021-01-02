from lightgbm import LGBMRegressor, plot_importance
import pandas as pd

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Get the model
def LGBM(q, X_train, Y_train, X_valid, Y_valid):
    model = LGBMRegressor(objective='quantile', alpha=q,
                         n_estimators=10000, bagging_fraction=0.7, learning_rate=0.027, subsample=0.7)
    model.fit(X_train, Y_train, eval_metric = ['quantile'], 
            eval_set=[(X_valid, Y_valid)], early_stopping_rounds=300, verbose=500)
    return model


def train_data(X_train, Y_train, X_valid, Y_valid):
    lst_models=[]
    for q in quantiles:
        print(q)    
        model = LGBM(q, X_train, Y_train, X_valid, Y_valid)
        lst_models.append(model)
    
    return lst_models

def test_data(models, X_test):
    df_pred = pd.DataFrame()
    for model in models:
        pred = pd.Series(model.predict(X_test).round(2))
        df_pred = pd.concat([df_pred, pred],axis=1)

    df_pred.columns=quantiles
    return df_pred

def plot_importance(models): 
    for model in models:
        plot_importance(model)