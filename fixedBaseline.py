#%%
import pandas as pd
import numpy as np
import os
import glob
import random
import warnings


def create_lag_feats(data, lags, cols):
    lag_cols = cols.copy()
    temp = data.copy()
    temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill') 
    temp['Target2'] = temp['TARGET'].shift(-96).fillna(method='ffill')  
    for col in cols:
        for lag in lags:
            temp[col + '_bfr_%s'%lag] = temp[col].shift(lag*48)  
            lag_cols.append(col + '_bfr_%s'%lag)
    return temp, lag_cols 

def preprocess_data(data, target_lags=[1], weather_lags=[1], is_train=True):
    temp = data.copy()
    if is_train==True:          
        temp, temp_lag_cols = create_lag_feats(temp, weather_lags, ['DHI', 'DNI', 'WS', 'RH', 'T'])
     
        return temp[['Hour'] + temp_lag_cols + ['Target1', 'Target2']].dropna()

    elif is_train==False:    
        temp, temp_lag_cols = create_lag_feats(temp, weather_lags, ['DHI', 'DNI', 'WS', 'RH', 'T'])
                              
        return temp[['Hour'] + temp_lag_cols].dropna()

train_data = pd.read_csv("./data/train/train.csv")
submission = pd.read_csv("./data/sample_submission.csv")
df_test = []
for i in range(81):
    file_path = './data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, target_lags=[1,2,3,4], weather_lags=[1,2,3,4], is_train=False).iloc[-48:]
    df_test.append(temp)
X_test = pd.concat(df_test)
df_train = preprocess_data(train_data, target_lags=[1,2,3,4], weather_lags=[1,2,3,4], is_train=True)
X_test.to_csv('./madeSample/xTest.csv', index = False)
df_train.to_csv('./madeSample/dfTrain.csv', index = False)
from sklearn.model_selection import train_test_split
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], test_size=0.3, random_state=0)
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

from lightgbm import LGBMRegressor
# Get the model and the predictions in (a) - (b)
def LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test):
    
    # (a) Modeling  
    model = LGBMRegressor(objective='quantile', alpha=q,
                         n_estimators=10000, learning_rate=0.02)                   
                         
                         
    model.fit(X_train, Y_train, eval_metric = ['quantile'],
          eval_set=[(X_valid, Y_valid)], early_stopping_rounds=300, verbose=500)

    # (b) Predictions
    pred = pd.Series(model.predict(X_test).round(2))
    return pred, model

# Target ¿¹Ãø
def predic_data(X_train, Y_train, X_valid, Y_valid, X_test):

    LGBM_models=[]
    LGBM_actual_pred = pd.DataFrame()

    for q in quantiles:
        print(q)
        pred , model = LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test)
        LGBM_models.append(model)
        LGBM_actual_pred = pd.concat([LGBM_actual_pred,pred],axis=1)

    LGBM_actual_pred.columns=quantiles
    
    return LGBM_models, LGBM_actual_pred

# Target1
models_1, results_1 = predic_data(X_train_1, Y_train_1, X_valid_1, Y_valid_1, X_test)

# Target2
models_2, results_2 = predic_data(X_train_2, Y_train_2, X_valid_2, Y_valid_2, X_test)
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values
submission.to_csv('./data/submission7.csv', index=False)
# %%
