#%%
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
trainCsv = pd.read_csv("./data/train/train.csv")
# DNI = 0 제거
list_zero = trainCsv[trainCsv["DHI"] == 0].index
removeZero=  trainCsv.drop(list_zero)
shuffled=removeZero.iloc[np.random.permutation(range(500))]

sns.regplot(x="DHI",y="TARGET", data=shuffled[:500])
# %%
# 0 제거 전
sns.distplot(trainCsv["DHI"])
# %%
# 0 제거 후
sns.distplot(shuffled["DHI"])