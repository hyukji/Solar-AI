#%%
import pandas as pd
import numpy as np
from modules.data_process import load_test, load_train
from sklearn.model_selection import train_test_split
from modules.LGBM import kFold_train_and_predict


trainData = load_train(days=3,cols=["DHI", "DNI", "WS", "RH", "T", "TARGET"])
testData = load_test(days=3,cols=["DHI", "DNI", "WS", "RH", "T", "TARGET"])

predictions_1 = kFold_train_and_predict(trainData.iloc[:, :-2], trainData.iloc[:, -2], testData)
predictions_2 = kFold_train_and_predict(trainData.iloc[:, :-2], trainData.iloc[:, -1], testData)

# submission.csv �ҷ�����
submission = pd.read_csv("./data/sample_submission.csv")
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = predictions_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = predictions_2.sort_index().values
# �����ϰ� ��� loss ���
submission.to_csv('./subs/20210105_1350.csv', index=False)
# print("===Loss mean: {0}".format(total_loss/18))
# %%
