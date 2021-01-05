import pandas as pd
import numpy as np
import random

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from data_module import get_train, get_test
from DL_model import Solar_Model, Solar_Dataset, Solar_loss, EarlyStopping

import torch
from torch import nn, optim 
from torch.utils.data import DataLoader, TensorDataset  

## parameter #######
epochs = 100
batch_size = 256
num_kfolds = 5
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
cons = 3  # cons = 연속적인 데이터 개수.
unit = 48  # unit = 시간간격 ex) 48 => 하루.
feature_num = 6 * cons # 맞나?????????????????
####################

X_train, Y_train= get_train(cons, unit)
# X_Test = get_test(cons, unit)

X_train = np.array(X_train.values)
Y_train = np.array(Y_train.values)
# X_Test = np.array(X_Test.values)

print(X_train.shape, Y_train[:, 0].reshape(-1, 1).shape, X_Test.shape)

X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(X_train, Y_train[:, 0].reshape(-1, 1), test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(X_train, Y_train[:, 1].reshape(-1, 1), test_size=0.3, random_state=0)


print(X_train_1.shape, Y_train_1.shape, X_valid_1.shape, Y_valid_1.shape)

train_1_Dataset = Solar_Dataset(X_train_1, Y_train_1)
valid_1_Dataset = Solar_Dataset(X_valid_1, Y_valid_1)
train_2_Dataset = Solar_Dataset(X_train_2, Y_train_2)
valid_2_Dataset = Solar_Dataset(X_valid_2, Y_valid_2)
# test_Dataset = TensorDataset(torch.tensor(X_Test, dtype=torch.float32))

loader_train_1 = DataLoader(train_1_Dataset, batch_size=batch_size, shuffle=True)
loader_valid_1 = DataLoader(valid_1_Dataset, batch_size=batch_size, shuffle=False)
loader_train_2 = DataLoader(train_2_Dataset, batch_size=batch_size, shuffle=True)
loader_valid_2 = DataLoader(valid_2_Dataset, batch_size=batch_size, shuffle=False)
# loader_test = DataLoader(test_Dataset, batch_size=batch_size, shuffle=False)

model = Solar_Model(18)
optimizer = optim.Adam(model.parameters(), lr=0.002)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98, last_epoch=-1)  # learning rate decay를 위한 learning rate scheduler를 정의함 (exponential learning rate decay, x0.98/epoch)

early_stopping = EarlyStopping(patience = 10, verbose= True)

print("training start")
for epoch in range(epochs):
    for mode in ['Train', 'Valid']:
        total_length = 0  # 데이터 개수를 추적
        running_loss = 0.  # loss를 추적

        if mode == 'Train':  # 학습 모드인 경우
            model.train()
            lr_scheduler.step()  # learning rate를 갱신함
            loader = loader_train_1
        else:  # valid 모드인 경우
            model.eval()
            loader = loader_valid_1


        for _, (x, y) in enumerate(loader):
            # print("x", x.shape)
            y_pred = model(x)
            loss = Solar_loss(y_pred, y, quantiles) 
            if mode == 'Train':  # 학습 모드인 경우
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                val_loss = loss.item()

        print('Epoch {}/{} Cost: {:.6f} mode: {}'.format(epoch, epochs, loss.item(), mode))
                
    if early_stopping.validate(val_loss):
        break

PATH = 'C:/Users/user/Desktop/ML대회/1. 태양광/model_data/'
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict()
}, PATH + 'all_1.tar')  # 여러 가지 값 저장, 학습 중 진행 상황 저장을 위해 epoch, loss 값 등 일반 scalar값 저장 가능
print("saved!")