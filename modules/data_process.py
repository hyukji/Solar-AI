import numpy as np
import pandas as pd
import time

# 전날 날씨 추가, 1,2일뒤의 Target 추가 함수
def loadPreviousDays(data, prevs, cols=['TARGET']):
    retCols = []
    retData = data.copy()
    retData["1DayAfter"] = retData['TARGET'].shift(-48, fill_value=np.nan)
    retData["2DayAfter"] = retData['TARGET'].shift(-96, fill_value=np.nan)
    for prev in prevs:
        for col in cols:
            tag= "{0}after{1}".format(prev,col)
            retData[tag] = retData[col].shift(prev*48, fill_value=np.nan)  
            retCols.append(tag)
    return retData, retCols

# 데이터 전처리 함수
def preprocessData(data, prevs=[1],cols=['TARGET'], isTrain=True):
    retData = data.copy()
    retData, retCols = loadPreviousDays(data, prevs, cols)
    if isTrain == True:
        return retData[retCols + ['1DayAfter', '2DayAfter']].dropna()
    else:
        return retData[retCols].dropna()


def load_train(days=1, cols=['TARGET']):
    trainCsv = pd.read_csv("./data/train/train.csv")
    trainData = preprocessData(trainCsv, prevs=range(days,-1,-1) ,cols=cols , isTrain=True)
    return trainData

def load_test(days=1, cols=['TARGET']):
    testList = []
    for i in range(81):
        file_path = './data/test/{0}.csv'.format(i)
        temp = pd.read_csv(file_path)
        temp = preprocessData(temp, prevs=range(days,-1,-1),cols=cols, isTrain=False).iloc[-48:]
        testList.append(temp)
    testData = pd.concat(testList)
    return testData

def save_trainData(data):
    now = time.localtime()
    data.to_csv(f"./data/{now.tm_year:02d}{now.tm_mon:02d}{now.tm_mday:02d}_{now.tm_hour:02d}{now.tm_min:02d}.csv")

def save_subs(subs):
    now = time.localtime()
    subs.to_csv(f"./subs/{now.tm_year:02d}{now.tm_mon:02d}{now.tm_mday:02d}_{now.tm_hour:02d}{now.tm_min:02d}.csv", index=False)

def delete_zero(data):
    temp = data.copy()
    list_zero = temp[temp["0afterDHI"] == 0].index
    temp = temp.drop(list_zero)
    return temp

def submission_zero(data, subs):
    temp = subs.copy()
    for i in range(81):
        for j in range(48):
            if(data.iloc[i*48+j]["0afterDHI"] == 0): 
                temp.iloc[i*48+j, 1:] = 0
    return temp