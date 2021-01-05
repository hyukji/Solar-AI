#%%
import pandas as pd
import numpy as np

from modules.data_process import load_test, load_train
from sklearn.model_selection import train_test_split
from modules.LGBM import train_model, predict_data


trainData = load_train(days=3,cols=["DHI", "DNI", "WS", "RH", "T", "TARGET"])
testData = load_test(days=3,cols=["DHI", "DNI", "WS", "RH", "T", "TARGET"])
# trainData.to_csv('./dfTrain.csv', index=False)
# # DHI가 0인 행 제거
# list_zero = trainData[trainData["0afterDHI"] == 0].index
# trainData =  trainData.drop(list_zero)

# train , test 분리
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(trainData.iloc[:, :-2], trainData.iloc[:, -2], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(trainData.iloc[:, :-2], trainData.iloc[:, -1], test_size=0.3, random_state=0)
#%%
# LGBM
totalLoss = 0
print("===FIRST DAY=")
models_1, loss_1 = train_model(X_train_1, Y_train_1, X_valid_1, Y_valid_1)
print("===SECOND DAY=")
models_2, loss_2 = train_model(X_train_2, Y_train_2, X_valid_2, Y_valid_2)
total_loss = loss_1 + loss_2
predictions_1 = predict_data(models_1, testData)
predictions_2 = predict_data(models_2, testData)

# submission.csv 불러오기
submission = pd.read_csv("./data/sample_submission.csv")
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = predictions_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = predictions_2.sort_index().values

# # DHI가 0인날 예측 0으로 채우기
# for i in range(81):
#     for j in range(48):
#         if(testData.iloc[i*48+j]["0afterDHI"] == 0): 
#             submission.iloc[i*48+j, 1:] = 0

# 저장하고 평균 loss 출력
submission.to_csv('./subs/20210104_1859.csv', index=False)
print("===Loss mean: {0}".format(total_loss/18))


#%%
#Feature importance
from lightgbm import plot_importance
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(6,6))
plot_importance(models_1[4],ax=ax)