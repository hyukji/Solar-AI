import pandas as pd
import numpy as np
import random
import copy

from sklearn.model_selection import train_test_split

from sl_data_module import get_train, get_test
from dh_data_module import load_train, delete_zero, load_test, load_change_train
from jh_data_module import get_train_total_data

from DL_model import LSTM_Model
from DL_model import LSTM_Model
from DL_module import Solar_Dataset, Solar_total_loss, EarlyStopping

import torch
from torch import nn, optim 
from torch.utils.data import DataLoader, TensorDataset  

from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold

import warnings
warnings.filterwarnings("ignore")


## parameter #######
epochs = 100
batch_size = 50
patience = 20
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
lr = 0.07

num_kfolds = 3
hidden_dim = 128
####################

trying_num = None
with open("trying_num.txt", 'r') as f:
    trying_num = f.read()

trying_num = str(int(trying_num) + 1)
with open("trying_num.txt", "w") as f:
    f.write(trying_num)


cols = ["Hour","DHI", "DNI", "WS", "RH", "T"]
history_date = 3
target_date = 1

X_train, Y_train = get_train_total_data(cols, history_date, target_date)
print(X_train.shape, Y_train.shape)


group_size = int(X_train.shape[0] / 3 )
groups = [0] * (group_size + 1) + [1] * (1+ group_size) + [2] * group_size
feature_num = X_train.shape[2] 

group_k_fold = GroupKFold(n_splits= num_kfolds)
# print("feature_num", feature_num, len(groups))

X = X_train
Y = Y_train
print("training start")

model = LSTM_Model(feature_num, hidden_dim, len(quantiles), 2, target_date)

for folder_num, (train_idx, valid_idx) in enumerate(group_k_fold.split(X, Y, groups)): 
    train_Dataset = Solar_Dataset(X[train_idx], Y[train_idx])
    valid_Dataset = Solar_Dataset(X[valid_idx], Y[valid_idx])
    loader_train = DataLoader(train_Dataset, batch_size=batch_size, shuffle=False)
    loader_valid = DataLoader(valid_Dataset, batch_size=batch_size, shuffle=False)


    optimizer = optim.Adam(model.parameters(), lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98, last_epoch=-1) 
    early_stopping = EarlyStopping(patience, verbose= True)

    for epoch in range(epochs):
        for mode in ['Train', 'Valid']:
            if mode == 'Train':  # 학습 모드인 경우
                model.train()
                lr_scheduler.step()  # learning rate를 갱신함
                loader = loader_train
            else:  # valid 모드인 경우
                model.eval()
                loader = loader_valid


            for _, (x, y) in enumerate(loader):
                y_pred = model(x)
                loss = Solar_total_loss(y_pred, y, quantiles) 
                optimizer.zero_grad()
                if mode == 'Train':  # 학습 모드인 경우
                    loss.backward()
                    optimizer.step()
                else:
                    val_loss = loss.item()

            print('f:{}, Epoch {}/{} Cost: {:.6f} mode: {}'.format(folder_num, epoch, epochs, loss.item(), mode))

            
        if early_stopping.validate(val_loss, model):
            PATH = 'C:/Users/user/Desktop/ML대회/1. 태양광/model_data/'
            torch.save({
            'model': early_stopping.best_model_wts(),
            'loss': early_stopping.best_loss(),
            }, PATH + f"{trying_num}_{folder_num}.tar")  # 모델 값 저장.
            print(f"{folder_num} saved!")
            break

