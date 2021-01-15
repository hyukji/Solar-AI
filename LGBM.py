# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from module.data import get_train, get_test, save_submission

cons = 4
unit = 48
removed_cols = ['Day', 'Minute']

# train_labels = pd.read_pickle('./train_labels.pkl')
# test_labels = pd.read_pickle('./test_labels.pkl')

df_train_x, df_train_y = get_train(cons, unit, removed_cols)
# df_train_x['season'] = train_labels
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(df_train_x, df_train_y.iloc[:, 0], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(df_train_x, df_train_y.iloc[:, 1], test_size=0.3, random_state=0)

X_test = get_test(cons, unit, removed_cols)
# X_test['season'] = test_labels
# %%
# get absolute test data from train data (For last cell)
X_train_1, ttx, Y_train_1, tty = train_test_split(X_train_1, Y_train_1, test_size=0.2, random_state=0)
# %%
from module.lgbm import test_data, train_data
from lightgbm import LGBMRegressor
# %%
def LGBM(q, X_train, Y_train, X_valid, Y_valid):
    model = LGBMRegressor(objective='quantile',
                         n_estimators=10000, bagging_fraction=0.7, learning_rate=0.027, subsample=0.7)
    model.fit(X_train, Y_train, eval_metric = ['quantile'], 
            eval_set=[(X_valid, Y_valid)], early_stopping_rounds=300, verbose=500)
    return model
    
def train_data(X_train, Y_train, X_valid, Y_valid):
    lst_models=[]
    # for q in quantiles:
        # print(q)    
    model = LGBM(0.1, X_train, Y_train, X_valid, Y_valid)
    lst_models.append(model)
    
    return lst_models


# Target1
models_1 = train_data(X_train_1, Y_train_1, X_valid_1, Y_valid_1)
# %%
df = models_1[0].booster_.trees_to_dataframe()
# %%
models_1 = train_data(X_train_1, Y_train_1, X_valid_1, Y_valid_1)
results_1 = test_data(models_1, X_test)
results_1[:48]

# Target2
models_2 = train_data(X_train_2, Y_train_2, X_valid_2, Y_valid_2)
# %%

# %%

# %%
results_2 = test_data(models_2, X_test)
results_2 = results_2.quantile([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1], axis=1).transpose()
results_2[:48]
# %%
results_1 = test_data(models_1, X_test)
results_1 = results_1.quantile([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1], axis=1).transpose()
results_1[:48]
# %%
results_1.to_pickle('./results_1.pkl')
results_2.to_pickle('./results_2.pkl')
# %%
save_submission(results_1, results_2, f'lgbm_quantile')
# plot_importance(models_1)

# %%
# deep learning model과 비교를 위해 동일한 함수 적용
from module.deep import tilted_loss

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# x, y의 순서가 같아야 함을 유의
res = 0
for q, model in zip(quantiles, models_1):
    pred = model.predict(ttx).round(2)
    tty = tty.reset_index(drop=True)
    a = tilted_loss(q, tty, pred)
    res += a
    print(a)
print('average', res/9)
# %%
