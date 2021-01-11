#%%
import pandas as pd
import numpy as np
from modules.data_process import load_test, load_train, save_subs,save_trainData
from modules.load_mvag import load_mAvg_train,load_mAvg_test
from modules.load_change import load_change_test, load_change_train
from modules.CatBM import kFold_train_and_predict



# trainData = load_train(days=3,cols=["DHI", "DNI", "WS", "RH", "T", "TARGET"])
# testData = load_test(days=3,cols=["DHI", "DNI", "WS", "RH", "T", "TARGET"])
trainData = load_mAvg_train(days=3, select=[1,3], cols=["DHI", "DNI", "WS", "RH", "T", "TARGET"],includeRaw=True)
testData = load_mAvg_test(days=3, select=[1,3], cols=["DHI", "DNI", "WS", "RH", "T", "TARGET"],includeRaw=True)
# avgData = add_moving_average(trainData, days=2, cols=["DHI", "DNI", "WS", "RH", "T", "TARGET"])
# save_trainData(avgData)
#%%
predictions_1,loss_1 = kFold_train_and_predict(trainData.iloc[:, :-2], trainData.iloc[:, -2], testData)
predictions_2,loss_2 = kFold_train_and_predict(trainData.iloc[:, :-2], trainData.iloc[:, -1], testData)

# submission.csv 불러오기
submission = pd.read_csv("./data/sample_submission.csv")
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = predictions_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = predictions_2.sort_index().values

# 저장하고 평균 loss 출력
save_subs(submission)
print("===Loss mean: {0}".format((loss_1+loss_2)/72))
# %%
