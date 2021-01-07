#%%
import pandas as pd
import numpy as np
from modules.data_process import save_subs
lgbmcsv = pd.read_csv("./submissions/20210103_2255.csv")
catcsv = pd.read_csv("./subs/20210106_1430.csv")
baggedCsv = lgbmcsv.copy()
baggedCsv.iloc[:,1:] = (lgbmcsv.iloc[:,1:] * 0.5) + (catcsv.iloc[:,1:] * 0.5)
save_subs(baggedCsv)
# %%
