import pandas as pd
import numpy as np
import random

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from sl_data_module import get_train, get_test
from dh_data_module import load_train, delete_zero, load_test, load_change_test
from jh_data_module import get_test_data
from DL_model import LSTM_Model

import torch
from torch import nn, optim 
from torch.utils.data import DataLoader, TensorDataset  

batch_size = 30
hidden_dim = 64
layers = 2

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


cols = ["Hour","DHI", "DNI", "WS", "RH", "T", "TARGET"]
history_date = 6
target_date = 1

X_Test = get_test_data(cols, history_date, target_date)
print(X_Test.shape)


feature_num = X_Test.shape[2] 
loader_test = DataLoader(torch.tensor(X_Test, dtype=torch.float32), batch_size=batch_size, shuffle=False)

print("test start feature_num", feature_num)

load_PATH = 'C:/Users/user/Desktop/ML대회/1. 태양광/model_data/'
save_PATH = 'C:/Users/user/Desktop/ML대회/1. 태양광/test_data/'

model = LSTM_Model(feature_num, hidden_dim, len(quantiles), layers, target_date)

submission = pd.read_csv('./test_data/merge_37_38.csv')
losses = []
sum_pred = []
zero_list = [0] * 9

for folder_num in range(num_kfolds):
    pred = []
    checkpoint = torch.load(load_PATH + f"{trying_num}_Day7_{folder_num}.tar")   # dict 불러오기
    model.load_state_dict(checkpoint['model'])
    loss = checkpoint['loss']
    losses.append(loss)

    model.eval()
    for b, x in enumerate(loader_test):
        y_pred = model(x).tolist()
        pred.extend([zero_list] * 10)
        pred.extend(y_pred)
        pred.extend([zero_list] * 8)

    pred = np.array(pred)
    if len(sum_pred) == 0:
        sum_pred = np.array(pred)
    else:
        sum_pred += np.array(pred)

    print(pred.shape)

submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = sum_pred
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] /= 3

np_losses = np.array(losses)
mean_loss = np.mean(np_losses)
print()
print(mean_loss)

submission.to_csv(save_PATH + f"all_{trying_num}_{round(mean_loss, 3)}.csv", index=False)

print("saved!")