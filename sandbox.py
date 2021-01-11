import pandas as pd
import numpy as np
import random

from dh_data_module import load_train, delete_zero, load_test, load_change_train

trainData = load_change_train(days=3, select=[1,3], cols=["DHI", "DNI", "WS", "RH", "T", "TARGET"], includeRaw=True)
mean_data = trainData.mean()
print(mean_data)
# submission.to_csv("mean_submission.csv")
