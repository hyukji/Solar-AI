import pandas as pd
import numpy as np
import random

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from sl_data_module import get_train, get_test
from dh_data_module import load_train, delete_zero, load_test, load_change_test
from DL_model import Day7_Model, Day8_Model, Solar_Dataset, Solar_loss, EarlyStopping

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

# X_Test = get_test(cons, unit)
trying_num = 53
# testData = load_test(days=3,cols=["DHI", "DNI", "WS", "RH", "T", "TARGET"])
testData = load_change_test(days=3, select=[1,3], cols=["DHI", "DNI", "WS", "RH", "T", "TARGET"],includeRaw=True)
X_Test = np.array(testData.values)

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
        print(loss)

np_losses = np.array(losses)
print()
print(np.mean(np_losses))

        