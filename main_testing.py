import pandas as pd
import numpy as np
import random

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from data_module import get_train, get_test
from DL_model import Day7_Model, Day8_Model, Solar_Dataset, Solar_loss, EarlyStopping

import torch
from torch import nn, optim 
from torch.utils.data import DataLoader, TensorDataset  


## parameter #######
epochs = 100
batch_size = 48
lr =0.002
num_kfolds = 5
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
cons = 3  # cons = 연속적인 데이터 개수.
unit = 48  # unit = 시간간격 ex) 48 => 하루.
feature_num = 6 * cons # 맞나?????????????????
####################


X_Test = get_test(cons, unit)
X_Test = np.array(X_Test.values)
print("X_Test", X_Test.shape)
loader_test = DataLoader(TensorDataset(torch.tensor(X_Test, dtype=torch.float32)), batch_size=batch_size, shuffle=False)

print("test start")
file_name = 'all_1_'
load_PATH = 'C:/Users/user/Desktop/ML대회/1. 태양광/model_data/'
save_PATH = 'C:/Users/user/Desktop/ML대회/1. 태양광/test_data/'
models = [Day7_Model(18), Day8_Model(18)]

preds = [[], []]   
for i, model in enumerate(models):
    pred = preds[i]
    optimizer = optim.Adam(model.parameters(), lr)

    checkpoint = torch.load(load_PATH + 'all_1_' + model._day() + '.tar')   # dict 불러오기
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    model.eval()
    for _, x in enumerate(loader_test):
        y_pred = model(x[0])
        pred.extend(y_pred.tolist())

    preds[i] = np.array(pred)

submission = pd.read_csv('./data/sample_submission.csv')
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = preds[0]
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = preds[1]
submission.to_csv(save_PATH + 'all_1_.csv', index=False)

print("saved!")