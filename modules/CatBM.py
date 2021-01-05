import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def CatBM(q, X_train, Y_train, X_valid, Y_valid):  
    model = CatBoostRegressor(
    iterations= 1000,
    learning_rate= 0.027,
    loss_function= f"Quantile:alpha={q}",
    subsample=0.7,
    devices='0:1'
    )
    model.fit(
        X=X_train,
        y=Y_train,
        verbose=500,
        early_stopping_rounds=300,
        eval_set=[(X_valid, Y_valid)]
    )
    loss = model.best_score_['validation'][f'Quantile:alpha={q}']
    return model, loss

def train_model(X_train, Y_train, X_valid, Y_valid):
    print("Training CatBoost Model .. ")
    models=[]
    totalLoss = 0
    for q in quantiles:
        print(f"quantile: {q}")
        model,loss = CatBM(q, X_train, Y_train, X_valid, Y_valid)
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