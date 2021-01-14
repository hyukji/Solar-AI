import pandas as pd
import numpy as np
import random
import copy

from sklearn.model_selection import train_test_split

from sl_data_module import get_train, get_test
from dh_data_module import load_train, delete_zero, load_test, load_change_train, save_trainData

from DL_model import Day7_Model, Day8_Model
from DL_module import Solar_Dataset, Solar_loss, EarlyStopping

import torch
from torch import nn, optim 
from torch.utils.data import DataLoader, TensorDataset  

from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold

import warnings
warnings.filterwarnings("ignore")


## parameter #######
epochs = 100
batch_size = 256
patience = 20
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
lr = 0.05

num_kfolds = 3
# one_year_data = 365 * 48
# groups = [0] * one_year_data + [1] * one_year_data + [2] * one_year_data


cons = 3  # cons = 연속적인 데이터 개수.
unit = 48  # unit = 시간간격 ex) 48 => 하루.
####################

trying_num = None
with open("trying_num.txt", 'r') as f:
    trying_num = f.read()

trying_num = str(int(trying_num) + 1)
with open("trying_num.txt", "w") as f:
    f.write(trying_num)


# X_train, Y_train= get_train(cons, unit)
# X_Test = get_test(cons, unit)

trainData = load_train(days=3,cols=["DHI", "DNI", "WS", "RH", "T", "TARGET"])
# trainData = load_change_train(days=3, select=[1,3], cols=["DHI", "DNI", "WS", "RH", "T", "TARGET"], includeRaw=True)
zero_trainData = delete_zero(trainData)
# save_trainData(zero_trainData)

X_train = np.array(zero_trainData.iloc[:, :-2].values)
Y_train = np.array(zero_trainData.iloc[:, -2:].values)

data_mean = X_train.mean(axis=0)
data_std = X_train.std(axis=0)
X_train = (X_train - data_mean) / data_std

print(X_train.shape, Y_train.shape)



group_size = int(X_train.shape[0] / 3 )
groups = [0] * (group_size + 1) + [1] * group_size + [2] * group_size
feature_num = X_train.shape[1] 

models = [Day7_Model(feature_num), Day8_Model(feature_num)]
group_k_fold = GroupKFold(n_splits= num_kfolds)


print("training start")
for i, model in enumerate(models):
    X = X_train
    Y = Y_train[:, i].reshape(-1, 1)
    # for idx in range(num_kfolds):
    #      X_shuffled, y_shuffled, groups_shuffled = shuffle(X, y, groups, random_state=i)
    
    for folder_num, (train_idx, valid_idx) in enumerate(group_k_fold.split(X, Y, groups)): 
        train_Dataset = Solar_Dataset(X[train_idx], Y[train_idx])
        valid_Dataset = Solar_Dataset(X[valid_idx], Y[valid_idx])
        loader_train = DataLoader(train_Dataset, batch_size=batch_size, shuffle=True)
        loader_valid = DataLoader(valid_Dataset, batch_size=batch_size, shuffle=False)


        model = Day7_Model(feature_num) if i == 0 else Day8_Model(feature_num)
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
                    # print("x", x.shape)
                    y_pred = model(x)
                    loss = Solar_loss(y_pred, y, quantiles) 
                    optimizer.zero_grad()
                    if mode == 'Train':  # 학습 모드인 경우
                        loss.backward()
                        optimizer.step()
                    else:
                        val_loss = loss.item()

                print('{} f:{}, Epoch {}/{} Cost: {:.6f} mode: {}'.format(model._day(),folder_num, epoch, epochs, loss.item(), mode))

            
            if early_stopping.validate(val_loss, model):
                PATH = 'C:/Users/user/Desktop/ML대회/1. 태양광/model_data/'
                torch.save({
                    'model': early_stopping.best_model_wts(),
                    'loss': early_stopping.best_loss(),
                }, PATH + f"{trying_num}_{model._day()}_{folder_num}.tar")  # 모델 값 저장.
                print(f"{model._day()}_{folder_num} saved!")
                break

