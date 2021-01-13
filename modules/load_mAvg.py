import numpy as np
import pandas as pd
from modules.data_process import preprocessData

# 전처리하고 이동평균 불러오기
# includeRaw 원본데이터를 유지하는가
def get_mAvg(data, days=1,select=[1],cols=['TARGET'], isTrain=True, includeRaw=False):
    temp = data.copy()
    temp = preprocessData(temp, prevs=range(days,-1,-1) ,cols=cols , isTrain=isTrain)
    temp = add_mAvg_data(temp, days,select,cols,isTrain,includeRaw)
    return temp

# 전처리가 되어있는 데이터에 이동평균 열 추가
def add_mAvg_data(data,days=1 ,select=[1],cols=['TARGET'], isTrain=True, includeRaw=False):
    temp = data.copy()
    retCols = []
    if includeRaw==True:
        if isTrain: 
            retCols=temp.columns[:-2].tolist()
        else:
            retCols=temp.columns.tolist()
    for col in cols:
        summ = temp[f"0after{col}"].copy()
        if includeRaw==False:
            retCols.append(f"0after{col}")
        for i in range(1,days+1):
            summ += temp[f"{i}after{col}"]
            if i in select:
                temp[f"{i}moveAvg{col}"] = summ/(i+1)
    for i in select:
        for col in cols:
            retCols.append(f"{i}moveAvg{col}")
    if isTrain == True:
        return temp[retCols + ['1DayAfter', '2DayAfter']]
    else:
        return temp[retCols]

def load_mAvg_train(days=1,select=[1],cols=['TARGET'],includeRaw=False):
    trainCsv = pd.read_csv("./data/train/train.csv")
    retData = get_mAvg(trainCsv,days,select,cols,True,includeRaw)
    return retData

def load_mAvg_test(days=1,select=[1], cols=['TARGET'],includeRaw=False):
    testList = []
    for i in range(81):
        file_path = './data/test/{0}.csv'.format(i)
        temp = pd.read_csv(file_path)
        temp = get_mAvg(temp, days,select,cols,False,includeRaw).iloc[-48:]
        testList.append(temp)
    testData = pd.concat(testList)
    return testData