#%%
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
trainCsv = pd.read_csv("./data/train/train.csv")
list_zero = trainCsv[trainCsv["DNI"] == 0].index
removeZero=  trainCsv.drop(list_zero)
shuffled=removeZero.iloc[np.random.permutation(range(500))].reset_index(drop=True)
sns.regplot(x="WS",y="TARGET", data=shuffled[:500])
# %%
sns.distplot(shuffled["DNI"])