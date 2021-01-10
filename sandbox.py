import pandas as pd
import numpy as np
import random

from dh_data_module import load_train, delete_zero, save_trainData

trainData = load_train(days=3,cols=["DHI", "DNI", "WS", "RH", "T", "TARGET"])
zero_trainData = delete_zero(trainData)
save_trainData(zero_trainData)