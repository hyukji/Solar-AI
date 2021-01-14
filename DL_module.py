import torch
from torch import nn
import copy
import torch.nn.functional as F
from torch.utils.data import Dataset

class Solar_Dataset(Dataset):
    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y

    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])

        return x, y


def Solar_loss(preds, y, quantiles):    
    losses = []
    for i, q in enumerate(quantiles): 

        error = y.squeeze() - preds[:, i]
        loss = torch.where(error > 0, error * q, error * (q - 1)).unsqueeze(1)
        losses.append(loss)
    
    loss_sum = torch.sum(torch.cat(losses, dim=1), dim=1)
    loss_mean = torch.mean(loss_sum)

    return loss_mean
    

class EarlyStopping(): # https://forensics.tistory.com/29 참조
    def __init__(self, patience = 10, verbose= True):
        self._step = 0
        self._loss = float('inf')
        self.patience  = patience
        self.verbose = verbose

        self.best_model_data = None

    def validate(self, loss, model):
        if self._loss < loss: #
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print(f'Training process is stopped early....')
                return True
        else:
            self._step = 0
            self._loss = loss
            self.best_model_data = copy.deepcopy(model.state_dict())
            print(f"step: {self._step}, loss: {self._loss}")

        return False

    def best_model_wts(self):
        return self.best_model_data

    def best_loss(self):
        return self._loss