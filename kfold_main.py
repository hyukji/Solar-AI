#%%
import pandas as pd
import numpy as np
from modules.data_process import load_test, load_train, save_csv
from modules.load_mAvg import load_mAvg_train,load_mAvg_test,add_mAvg_data
from modules.load_change import load_change_test, load_change_train,add_change_data
from modules.CatBM import kFold_train_and_predict


# trainData = load_change_train(days=3, select=[1,2,3], cols=["DHI", "DNI", "WS", "RH", "T", "TARGET"],includeRaw=True,fromZero=True)
# testData = load_change_test(days=3, select=[1,2,3], cols=["DHI", "DNI", "WS", "RH", "T", "TARGET"],includeRaw=True,fromZero=True)
cols = ["DHI", "DNI", "WS", "RH", "T", "TARGET"]

trainData = load_train(days=3, cols=cols)
changeTrain = add_change_data(trainData,select=[1,3],cols=cols,isTrain=True,includeRaw=True,fromZero=True)
moveTrain = add_mAvg_data(changeTrain,days=3,select=[1,3],cols=cols,isTrain=True,includeRaw=True)

testData = load_test(days=3, cols=cols)
changeTest = add_change_data(testData,select=[1,3],cols=cols,isTrain=False,includeRaw=True,fromZero=True)
moveTest = add_mAvg_data(changeTest,days=3,select=[1,3],cols=cols,isTrain=False,includeRaw=True)
save_csv(moveTest)
#%%
predictions_1,loss_1 = kFold_train_and_predict(trainData.iloc[:, :-2], trainData.iloc[:, -2], testData)
predictions_2,loss_2 = kFold_train_and_predict(trainData.iloc[:, :-2], trainData.iloc[:, -1], testData)

# submission.csv 불러오기
submission = pd.read_csv("./data/sample_submission.csv")
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = predictions_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = predictions_2.sort_index().values

# 저장하고 평균 loss 출력
save_csv(submission,isSub=True)
print("===Loss mean: {0}".format((loss_1+loss_2)/54))
# %%
