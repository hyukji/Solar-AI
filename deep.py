# %%
import pandas as pd
import numpy as np

from module.data import get_train, get_test, save_submission
# %%
cons = 96
unit = 1

df_train_x, df_train_y = get_train(cons, unit, save=False)
X_test = get_test(cons, unit, save=False)

# Normalize
mean = df_train_x.mean(axis=0)
std = df_train_x.std(axis=0)
df_train_x = (df_train_x - mean) / std
X_test = (X_test - mean) / std

from sklearn.model_selection import train_test_split
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(df_train_x, df_train_y.iloc[:, 0], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(df_train_x, df_train_y.iloc[:, 1], test_size=0.3, random_state=0)
# %%
from module.deep import train_data, test_data
dims = [576, 192, 60]

# Target1
models_1 = train_data(dims, X_train_1, Y_train_1, X_valid_1, Y_valid_1)
results_1 = test_data(models_1, X_test)
results_1[:48]

# Target2
models_2 = train_data(dims, X_train_2, Y_train_2, X_valid_2, Y_valid_2)
results_2 = test_data(models_2, X_test)
results_2[:48]
# %%
save_submission(results_1, results_2, f'deep_{cons}_{unit}')
# %%
# deep learning model과 비교를 위해 동일한 함수 적용
from module.deep import tilted_loss
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

actual = pd.DataFrame()
for q, model in zip(quantiles, models_1):
    pred = model.predict(X_valid_1)
    pred = np.squeeze(pred)
    pred = pd.Series(pred.round(2))
    Y_valid_1 = Y_valid_1.reset_index(drop=True)
    print(tilted_loss(q, Y_valid_1, pred))