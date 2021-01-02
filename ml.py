#%%
import pandas as pd
import numpy as np
import os
import random
import warnings
from sklearn.model_selection import train_test_split

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
def preprocessData(data, prevs=[1], isTrain=True):
    retData = data.copy()
    retData, retCols = loadPreviousDays(data, prevs, ['DHI', 'DNI', 'WS', 'RH', 'T'])
    if isTrain == True:
        return retData[['Hour'] + retCols + ['1DayAfter', '2DayAfter']].dropna()
    else:
        return retData[['Hour'] + retCols].dropna()

# LOAD DATA
trainCsv = pd.read_csv("./data/train/train.csv")
trainData = preprocessData(trainCsv, prevs=[1,2,3,4], isTrain=True)
list_zero = trainData[trainData["DHI"] == 0].index
trainData =  trainData.drop(list_zero)

testList = []
for i in range(81):
    file_path = './data/test/{0}.csv'.format(i)
    temp = pd.read_csv(file_path)
    temp = preprocessData(temp, prevs=[1,2,3,4], isTrain=False).iloc[-48:]
    testList.append(temp)
testData = pd.concat(testList)


xTrain1, xValid1, yTrain1, yValid1 = train_test_split(trainData.iloc[:, :-2], trainData.iloc[:, -2], test_size=0.3, random_state=0)
xTrain2, xValid2, yTrain2, yValid2 = train_test_split(trainData.iloc[:, :-2], trainData.iloc[:, -1], test_size=0.3, random_state=0)
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
trainData.head()
#%%
from lightgbm import LGBMRegressor
lgb_params = {
    'n_estimators':14000,
    'learning_rate':0.02
}
totalScore = 0
predictionOneDay = pd.DataFrame()
predictionTwoDay = pd.DataFrame()
for q in quantiles:
    print("===First Day quantile: {0}".format(q))
    model = LGBMRegressor(objective='quantile',alpha=q, **lgb_params)
    model.fit(xTrain1, yTrain1, eval_metric=['quantile'], eval_set=[(xValid1, yValid1)]
            ,early_stopping_rounds=300, verbose=500
            )
    pred = pd.Series(model.predict(testData).round(2))
    predictionOneDay = pd.concat([predictionOneDay,pred], axis=1)
    totalScore += model.best_score_['valid_0']['quantile']
predictionOneDay.columns = quantiles
for q in quantiles:
    print("===Second Day quantile: {0}".format(q))
    model = LGBMRegressor(objective='quantile',alpha=q, **lgb_params)
    model.fit(xTrain2, yTrain2, eval_metric=['quantile'], eval_set=[(xValid2, yValid2)]
            ,early_stopping_rounds=300, verbose=500
            )
    pred = pd.Series(model.predict(testData).round(2))
    predictionTwoDay = pd.concat([predictionTwoDay,pred], axis=1)
    totalScore += model.best_score_['valid_0']['quantile']
predictionTwoDay.columns = quantiles
submission = pd.read_csv("./data/sample_submission.csv")
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = predictionOneDay.values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = predictionTwoDay.values
for i in range(81):
    for j in range(48):
        if(testData.iloc[i*48+j]["DNI"] == 0 and testData.iloc[i*48+j]["DHI"] == 0): 
            submission.iloc[i*48+j, 1:] = 0
            # submission.loc["{0}.csv_Day7_{1}h{2:02d}m".format(i,hour,minute), "q_0.1":] = 0
submission.to_csv('./subs/WithHour.csv', index=False)
print("SCORE MEAN: {0}".format(totalScore/18))
print("END")

#%%