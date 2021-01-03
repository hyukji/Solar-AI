#%%
import pandas as pd
import numpy as np

from modules.data_process import load_test, load_train
from sklearn.model_selection import train_test_split
trainData = load_train(days=3,cols=["DHI", "DNI", "WS", "RH", "T", "TARGET"])
testData = load_test(days=3,cols=["DHI", "DNI", "WS", "RH", "T", "TARGET"])
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(trainData.iloc[:, :-2], trainData.iloc[:, -2], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(trainData.iloc[:, :-2], trainData.iloc[:, -1], test_size=0.3, random_state=0)
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#%%
from modules.CatBM import train_model, predict_data
totalLoss = 0
print("===FIRST DAY=")
models_1, loss_1 = train_model(X_train_1, Y_train_1, X_valid_1, Y_valid_1)
print("===SECOND DAY=")
models_2, loss_2 = train_model(X_train_2, Y_train_2, X_valid_2, Y_valid_2)
total_loss = loss_1 + loss_2
predictions_1 = predict_data(models_1, testData)
predictions_2 = predict_data(models_2, testData)

# submission.csv �ҷ�����
submission = pd.read_csv("./data/sample_submission.csv")
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = predictions_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = predictions_2.sort_index().values

# # DHI�� 0�γ� ���� 0���� ä���
# for i in range(81):
#     for j in range(48):
#         if(testData.iloc[i*48+j]["DHI"] == 0): 
#             submission.iloc[i*48+j, 1:] = 0

# �����ϰ� ��� loss ���
submission.to_csv('./subs/20210104_0012.csv', index=False)
print("===Loss mean: {0}".format(total_loss/18))