#%%
import pandas as pd
import numpy as np
from modules.data_process import save_subs
bag1 = pd.read_csv("./subs/20210106_1846.csv")
bag2 = pd.read_csv("./subs/20210108_1638.csv")
bag3 = pd.read_csv("./subs/20210111_1920.csv")
baggedCsv = bag1.copy()
baggedCsv.iloc[:,1:] = (bag1.iloc[:,1:] * 0.4) + (bag2.iloc[:,1:] * 0.2) + (bag3.iloc[:,1:] * 0.4)
save_subs(baggedCsv)
# %%
