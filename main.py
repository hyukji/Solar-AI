# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from module.data import get_train, get_test, save_submission

cons = 2
unit = 48

df_train_x, df_train_y = get_train(cons, unit, save=False)
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(df_train_x, df_train_y.iloc[:, 0], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(df_train_x, df_train_y.iloc[:, 1], test_size=0.3, random_state=0)

X_test = get_test(cons, unit, save=False)

# %%
from module.lgbm import train_data, test_data
# Target1
models_1 = train_data(X_train_1, Y_train_1, X_valid_1, Y_valid_1)
results_1 = test_data(models_1, X_test)
results_1[:48]

# Target2
models_2 = train_data(X_train_2, Y_train_2, X_valid_2, Y_valid_2)
results_2 = test_data(models_2, X_test)
results_2[:48]
# %%
save_submission(results_1, results_2, f'lgbm_{cons}_{unit}')
# plot_importance(models_1)

# %%
# deep learning model과 비교를 위해 동일한 함수 적용
from module.deep import tilted_loss

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# x, y의 순서가 같아야 함을 유의
for q, model in zip(quantiles, models_1):
    pred = model.predict(X_valid_1).round(2)
    Y_valid_1 = Y_valid_1.reset_index(drop=True)
    print(tilted_loss(q, Y_valid_1, pred))