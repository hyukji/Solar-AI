import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
lgb_params = {
    'n_estimators':10000,
    'learning_rate':0.027,
    'bagging_fraction':0.7,
    'subsample':0.7
}

def LGBM(q, X_train, Y_train, X_valid, Y_valid):  
    model = LGBMRegressor(objective='quantile', alpha=q, **lgb_params)                        
    model.fit(X_train, Y_train, eval_metric = ['quantile'],
          eval_set=[(X_valid, Y_valid)], early_stopping_rounds=300, verbose=500)
    loss = model.best_score_['valid_0']['quantile']
    return model, loss

def train_model(X_train, Y_train, X_valid, Y_valid):
    models=[]
    totalLoss = 0
    for q in quantiles:
        print(q)
        model,loss = LGBM(q, X_train, Y_train, X_valid, Y_valid)
        models.append(model)
        totalLoss += loss
    return models, totalLoss

def predict_data(models, X_test):
    predictions = pd.DataFrame()
    for model in models:
        pred = pd.Series(model.predict(X_test).round(2))
        predictions = pd.concat([predictions,pred],axis=1)
    predictions.columns = quantiles
    return predictions