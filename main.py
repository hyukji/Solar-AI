#%%
import pandas as pd
import numpy as np
import os
import random
import warnings
from sklearn.model_selection import train_test_split

# 전날 날씨 추가, 1,2일뒤의 Target 추가 함수
def loadPreviousDays(data, prevs, cols):
    retCols = cols.copy()
    retData = data.copy()
    retData["1DayAfter"] = retData['TARGET'].shift(-48)
    retData["2DayAfter"] = retData['TARGET'].shift(-96)
    for prev in prevs:
        for col in cols:
            tag= "{0}before{1}".format(prev,col)
            retData[tag] = retData[col].shift(prev*48)  
            retCols.append(tag)
    return retData, retCols

# 데이터 전처리 함수
def preprocessData(data, prevs=[1], isTrain=True):
    retData = data.copy()
    retData, retCols = loadPreviousDays(data, prevs, ['DHI', 'DNI', 'WS', 'RH', 'T'])
    if isTrain == True:
        return retData[['Hour'] + retCols + ['1DayAfter', '2DayAfter']].dropna()
    else:
        return retData[['Hour'] + retCols].dropna()

# csv 불러오기
trainCsv = pd.read_csv("./data/train/train.csv")
trainData = preprocessData(trainCsv, prevs=[1,2,3,4], isTrain=True)

# TEST 데이터 불러오기
testList = []
for i in range(81):
    file_path = './data/test/{0}.csv'.format(i)
    temp = pd.read_csv(file_path)
    temp = preprocessData(temp, prevs=[1,2,3,4], isTrain=False).iloc[-48:]
    testList.append(temp)
testData = pd.concat(testList)

# DHI가 0인 행 제거
list_zero = trainData[trainData["DHI"] == 0].index
trainData =  trainData.drop(list_zero)

# train , test 분리
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(trainData.iloc[:, :-2], trainData.iloc[:, -2], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(trainData.iloc[:, :-2], trainData.iloc[:, -1], test_size=0.3, random_state=0)
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
trainData.head()
#%%
from lightgbm import LGBMRegressor

totalLoss = 0
lgb_params = {
    'n_estimators':12000,
    'learning_rate':0.01
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
        pred = pd.Series(model.predict(testData).round(2))
        predictions = pd.concat([predictions,pred],axis=1)
    predictions.columns = quantiles
    return predictions

print("===FIRST DAY=")
models_1, loss_1 = train_model(X_train_1, Y_train_1, X_valid_1, Y_valid_1)
print("===SECOND DAY=")
models_2, loss_2 = train_model(X_train_2, Y_train_2, X_valid_2, Y_valid_2)
total_loss = loss_1 + loss_2
predictions_1 = predict_data(models_1, testData)
predictions_2 = predict_data(models_2, testData)

# submission.csv 불러오기
submission = pd.read_csv("./data/sample_submission.csv")
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = predictions_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = predictions_2.sort_index().values

# DHI가 0인날 예측 0으로 채우기
for i in range(81):
    for j in range(48):
        if(testData.iloc[i*48+j]["DHI"] == 0): 
            submission.iloc[i*48+j, 1:] = 0

# 저장하고 평균 loss 출력
submission.to_csv('./subs/20210102_1558.csv', index=False)
print("===Loss mean: {0}".format(total_loss/18))

#%%