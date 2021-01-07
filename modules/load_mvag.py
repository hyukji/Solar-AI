import numpy as np
import pandas as pd
from modules.data_process import preprocessData

def moving_average(data, days=1,select=[1],cols=['TARGET'], isTrain=True):
    temp = data.copy()
    temp = preprocessData(temp, prevs=range(days,-1,-1) ,cols=cols , isTrain=isTrain)
    retCols = []
    for col in cols:
        summ = temp[f"0after{col}"].copy()
        retCols.append(f"0after{col}")
        for i in range(1,days+1):
            summ += temp[f"{i}after{col}"]
            temp[f"{i}moveAvg{col}"] = summ/(i+1)
    for i in select:
        for col in cols:
            retCols.append(f"{i}moveAvg{col}")
    if isTrain == True:
        return temp[retCols + ['1DayAfter', '2DayAfter']]
    else:
        return temp[retCols]

def load_mAvg_train(days=1,select=[1],cols=['TARGET']):
    trainCsv = pd.read_csv("./data/train/train.csv")
    retData = moving_average(trainCsv,days,select,cols)
    return retData

def load_mAvg_test(days=1,select=[1], cols=['TARGET']):
    testList = []
    for i in range(81):
        file_path = './data/test/{0}.csv'.format(i)
        temp = pd.read_csv(file_path)
        temp = moving_average(temp, days,select,cols=cols, isTrain=False).iloc[-48:]
        testList.append(temp)
    testData = pd.concat(testList)
    return testData