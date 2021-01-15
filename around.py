# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from module.data import get_train, get_test, save_submission
from module.data import concat_data

removed_cols = ['Day', 'Minute']
cons = 4
unit = 48
additional = (2, -1) # cons and unit
df_train_x, df_train_y = get_train(cons, unit, removed_cols, additional)

X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(df_train_x, df_train_y.iloc[:, 0], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(df_train_x, df_train_y.iloc[:, 1], test_size=0.3, random_state=0)

X_test = get_test(cons, unit, removed_cols, additional)
# %%
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# quantiles = [0.4, 0.5, 0.6]
from module.lgbm import train_data, test_data
# %%
# Target1
models_1 = train_data(X_train_1, Y_train_1, X_valid_1, Y_valid_1)
results_1 = test_data(models_1, X_test)
results_1[:48]

# Target2
models_2 = train_data(X_train_2, Y_train_2, X_valid_2, Y_valid_2)
results_2 = test_data(models_2, X_test)
results_2[:48]
# %%
save_submission(results_1, results_2, f'around')
# plot_importance(models_1)