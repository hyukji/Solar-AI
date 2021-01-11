import numpy as np
import pandas as pd
from modules.data_process import preprocessData

def get_change(data, days=1,select=[1],cols=['TARGET'], isTrain=True, includeRaw=False):
    temp = data.copy()
    temp = preprocessData(temp, prevs=range(days,-1,-1) ,cols=cols , isTrain=isTrain)
    retCols = []
    for col in cols:
        retCols.append(f"0after{col}")
        for i in range(1,days+1):
            if includeRaw==True:
                retCols.append(f"{i}after{col}")
            temp[f"{i}diff{col}"] = temp[f"{i}after{col}"] - temp[f"{i-1}after{col}"]
    for i in select:
        for col in cols:
            retCols.append(f"{i}diff{col}")
    if isTrain == True:
        return temp[retCols + ['1DayAfter', '2DayAfter']]
    else:
        return temp[retCols]

def load_change_train(days=1,select=[1],cols=['TARGET'],includeRaw=False):
    trainCsv = pd.read_csv("./data/train/train.csv")
    retData = get_change(trainCsv,days,select,cols,True,includeRaw)
    return retData

def load_change_test(days=1,select=[1], cols=['TARGET'],includeRaw=False):
    testList = []
    for i in range(81):
        file_path = './data/test/{0}.csv'.format(i)
        temp = pd.read_csv(file_path)
        temp = get_change(temp, days,select,cols,False,includeRaw).iloc[-48:]
        testList.append(temp)
    testData = pd.concat(testList)
    return testData