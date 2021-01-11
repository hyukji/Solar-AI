import pandas as pd
import numpy as np


df1 = pd.read_csv("./test_data/all_37_.csv")
df2 = pd.read_csv("./test_data/all_38_.csv")

df1.iloc[:,1:] += df2.iloc[:,1:]
df1.iloc[:,1:] /= 2

df1.to_csv("./test_data/merge_37_38.csv", index=False)