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

checkpoint = torch.load(PATH + 'all_1.tar')   # dict 불러오기
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

X_Test = get_test(cons, unit)
X_Test = np.array(X_Test.values)
test_Dataset = TensorDataset(torch.tensor(X_Test, dtype=torch.float32))
loader_test = DataLoader(test_Dataset, batch_size=batch_size, shuffle=False)


model = Solar_Model(18)
optimizer = optim.Adam(model.parameters(), lr=0.002)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98, last_epoch=-1) 