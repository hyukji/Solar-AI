import pandas as pd
import numpy as np
import random

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from sl_data_module import get_train, get_test
from dh_data_module import load_train, delete_zero, load_test, load_change_test
from DL_model import Day7_Model, Day8_Model
from DL_module import Solar_Dataset, Solar_loss, EarlyStopping

import torch
from torch import nn, optim 
from torch.utils.data import DataLoader, TensorDataset  

batch_size = 48

## parameter #######
num_kfolds = 3
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


cons = 3  # cons = 연속적인 데이터 개수.
unit = 48  # unit = 시간간격 ex) 48 => 하루.
####################

trying_num = None
with open("trying_num.txt", 'r') as f:
    trying_num = f.read()

with open("trying_num.txt", "w") as f:
    f.write(trying_num)

trying_num = 42
# X_Test = get_test(cons, unit)

testData = load_test(days=3,cols=["DHI", "DNI", "WS", "RH", "T", "TARGET"])
# testData = load_change_test(days=3, select=[1,3], cols=["DHI", "DNI", "WS", "RH", "T", "TARGET"],includeRaw=True)
X_Test = np.array(testData.values)

# data_mean = X_Test.mean(axis=0)
# data_std = X_Test.std(axis=0)
# X_Test = (X_Test - data_mean) / data_std

print("X_Test", X_Test.shape)

feature_num = X_Test.shape[1] 

loader_test = DataLoader(TensorDataset(torch.tensor(X_Test, dtype=torch.float32)), batch_size=batch_size, shuffle=False)

print("test start")

load_PATH = 'C:/Users/user/Desktop/ML대회/1. 태양광/model_data/'
save_PATH = 'C:/Users/user/Desktop/ML대회/1. 태양광/test_data/'
models = [Day7_Model(feature_num), Day8_Model(feature_num)]

submission = pd.read_csv('./data/sample_submission.csv')
losses = []
for folder_num in range(num_kfolds):
    preds = [[], []]  
    for m, model in enumerate(models):
        pred = preds[m]
    
        checkpoint = torch.load(load_PATH + f"{trying_num}_{model._day()}_{folder_num}.tar")   # dict 불러오기
        model.load_state_dict(checkpoint['model'])
        loss = checkpoint['loss']
        losses.append(loss)

        model.eval()
        for b, x in enumerate(loader_test):
            y_pred = model(x[0]).tolist()
            for i in range(48): # 6일의 target == 0이면 7,8일의 target = 0
                if X_Test[(48 * b) + i][-1] == 0:
                    y_pred[i] = [0] * 9
            pred.extend(y_pred)

        preds[m] = np.array(pred)
        print(preds[m].shape, loss)

    submission.loc[submission.id.str.contains("Day7"), "q_0.1":] += preds[0]
    submission.loc[submission.id.str.contains("Day8"), "q_0.1":] += preds[1]
    
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] /= 3
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] /= 3

np_losses = np.array(losses)
mean_loss = np.mean(np_losses)
print()
print(mean_loss)

submission.to_csv(save_PATH + f"all_{trying_num}_{round(mean_loss, 3)}.csv", index=False)

print("saved!")
