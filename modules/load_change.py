import numpy as np
import pandas as pd
from modules.data_process import preprocessData

# 전처리하고 변화량 불러오기
# includeRaw 원본데이터를 유지하는가
# fromZero 변화량 기준이 0일(오늘)부터인가
def get_change(data, days=1,select=[1],cols=['TARGET'], isTrain=True, includeRaw=False, fromZero=False):
    temp = data.copy()
    temp = preprocessData(temp,unit=1 ,prevs=range(days,-1,-1) ,cols=cols , isTrain=isTrain)
    retCols = []
    for col in cols:
        retCols.append(f"0after{col}")
        for i in range(1,days+1):
            if includeRaw==True:
                retCols.append(f"{i}after{col}")
            if fromZero==False:
                temp[f"{i}diff{col}"] = temp[f"{i}after{col}"] - temp[f"{i-1}after{col}"]
            else:
                temp[f"{i}diff{col}"] = temp[f"{i}after{col}"] - temp[f"0after{col}"]
    for i in select:
        for col in cols:
            retCols.append(f"{i}diff{col}")
    if isTrain == True:
        return temp[retCols + ['1DayAfter', '2DayAfter']]
    else:
        return temp[retCols]

# 전처리가 되어있는 데이터에 변화량 열 추가
def add_change_data(data,select=[1],cols=['TARGET'],isTrain=True ,includeRaw=False, fromZero=False):
    temp = data.copy()
    retCols = []
    if includeRaw==True:
        if isTrain: 
            retCols=temp.columns[:-2].tolist()
        else:
            retCols=temp.columns.tolist()
    for col in cols:
        if includeRaw==False:
            retCols.append(f"0after{col}")
        for i in select:
            if fromZero==False:
                temp[f"{i}diff{col}"] = temp[f"{i}after{col}"] - temp[f"{i-1}after{col}"]
            else:
                temp[f"{i}diff{col}"] = temp[f"{i}after{col}"] - temp[f"0after{col}"]
    for i in select:
        for col in cols:
            retCols.append(f"{i}diff{col}")
    if isTrain == True:
        return temp[retCols + ['1DayAfter', '2DayAfter']]
    else:
        return temp[retCols]

def load_change_train(days=1,select=[1],cols=['TARGET'],includeRaw=False, fromZero=False):
    trainCsv = pd.read_csv("./data/train/train.csv")
    retData = get_change(trainCsv,days,select,cols,True,includeRaw,fromZero)
    return retData

def load_change_test(days=1,select=[1], cols=['TARGET'],includeRaw=False, fromZero=False):
    testList = []
    for i in range(81):
        file_path = './data/test/{0}.csv'.format(i)
        temp = pd.read_csv(file_path)
        temp = get_change(temp, days,select,cols,False,includeRaw,fromZero).iloc[-48:]
        testList.append(temp)
    testData = pd.concat(testList)
    return testData