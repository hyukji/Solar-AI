# %%
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
# %%
removed_cols = [ 'Minute']

def add_future_feats(data, after_days, cols):
    # day: 1~
    temp = data.copy()
    new_cols = []
    for col in cols:
        for i, day in enumerate(after_days):
            new_col = f'{col}_after_{i+1}'
            temp[new_col] = temp[col].shift(-(day*48), fill_value = np.nan)
            new_cols.append(new_col)
    return temp

def concat_days(data, n_days=2): # num of consecutive days
    temp = data.copy()
    res = data.copy()
    cols_name = data.columns
    res.columns = [f'{col}_{0}' for col in cols_name]
    for day in range(1, n_days):
        one_day = temp.shift(-day*48, fill_value = np.nan)
        one_day.columns = [f'{col}_{day}' for col in cols_name]
        res = pd.concat([res, one_day],axis=1)
    return res

def preprocess_data(data, is_train=True):
    temp = data.copy()
    # temp['GHI'] = temp['DHI'] + temp['DNI']
    temp = temp.drop(removed_cols, axis='columns')

    if is_train==True:
        after_days = [1, 2]
        cols = ['TARGET_3']

        temp = concat_days(temp, n_days=4)
        temp = add_future_feats(temp, after_days, cols)
     
        temp = temp.dropna()
        return temp

    elif is_train==False:

        temp = get_days(temp, n_days=4)
        return temp

# for test
def get_days(data, n_days): # day: 0~6
    assert data.shape[0] == 336 # 48*7
    temp = data.copy()
    res = pd.DataFrame()
    for idx, day in enumerate(range(6, 6-n_days, -1)):
        one_day = temp.iloc[day*48:(day+1)*48]
        one_day.reset_index(drop=True, inplace=True)
        cols_name = one_day.columns
        one_day.columns = [f'{col}_{idx}' for col in cols_name]
        res = pd.concat([res, one_day],axis=1)
    return res

# %% finding quantile of train data
train = pd.read_csv('./data/train/train.csv')
df_train = preprocess_data(train, is_train=True)
grouped= df_train[[f'TARGET_{i}' for i in range(4)]].groupby(df_train['Day_1'])
y = grouped.mean()
y = y.mean(axis=1)
x = range(1087)

fig = plt.figure(figsize=(50,10))
ax = fig.add_subplot(1, 1, 1)

ax.grid(color='#BDBDBD', linestyle='-', linewidth=2)
ax.plot(x, y)
ax.xaxis.set_ticks(range(40, 2))
# ax.yaxis.set_ticks(range(1080, 50))
plt.savefig('hh.png')
plt.show()

y.quantile([0, 0.33, 0.66, 0.99, 1])
# ~ 13.834414 ~ 21.438621 ~ (7일의 평균이 세 구간 중 어디에 속하나? 모델 3개)
# %%
train = pd.read_csv('./data/train/train.csv')
df_train = preprocess_data(train, is_train=True)
grouped= df_train[[f'TARGET_{i}' for i in range(4)]].groupby(df_train['Day_0'])
day_mean = grouped.mean()
week_mean = day_mean.mean(axis=1)

# 데이터 사이즈 복원하기
temp = pd.Series(index = range(df_train.shape[0]))
for i in range(week_mean.size):
    temp.loc[i*48] = week_mean[i]
temp = temp.fillna(method='ffill')

# 데이터 분류하기
winter = temp <= 13.8344
fall = (13.8344 < temp) & (temp <= 21.4386)
summer = temp > 21.4386

w_train = df_train[winter]
f_train = df_train[fall]
s_train = df_train[summer]
#%%
from sklearn.model_selection import train_test_split
Xw_train_1, Xw_valid_1, Yw_train_1, Yw_valid_1 = train_test_split(w_train.iloc[:, :-2], w_train.iloc[:, -2], test_size=0.3, random_state=0)
Xw_train_2, Xw_valid_2, Yw_train_2, Yw_valid_2 = train_test_split(w_train.iloc[:, :-2], w_train.iloc[:, -1], test_size=0.3, random_state=0)

Xf_train_1, Xf_valid_1, Yf_train_1, Yf_valid_1 = train_test_split(f_train.iloc[:, :-2], f_train.iloc[:, -2], test_size=0.3, random_state=0)
Xf_train_2, Xf_valid_2, Yf_train_2, Yf_valid_2 = train_test_split(f_train.iloc[:, :-2], f_train.iloc[:, -1], test_size=0.3, random_state=0)

Xs_train_1, Xs_valid_1, Ys_train_1, Ys_valid_1 = train_test_split(s_train.iloc[:, :-2], s_train.iloc[:, -2], test_size=0.3, random_state=0)
Xs_train_2, Xs_valid_2, Ys_train_2, Ys_valid_2 = train_test_split(s_train.iloc[:, :-2], s_train.iloc[:, -1], test_size=0.3, random_state=0)
# %%
df_tests = []
means = []
for i in range(81):
    file_path = './data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    means.append( temp[[f'TARGET_{i}' for i in range(4)]].mean(axis=0).mean())
    df_tests.append(temp)

df_test = pd.concat(df_tests)
X_test = df_test
week_mean = pd.Series(means)

# 데이터 사이즈 복원하기
temp = pd.Series(index = range(df_test.shape[0]))
for i in range(week_mean.size):
    temp.loc[i*48] = week_mean[i]
temp = temp.fillna(method='ffill')

# 데이터 분류하기
winter = temp <= 13.8344
fall = (13.8344 < temp) & (temp <= 21.4386)
summer = temp > 21.4386
season = pd.Series(index=range(df_test.shape[0]))
season = np.where(winter, 'winter', season)
season = np.where(fall, 'fall', season)
season = np.where(summer, 'summer', season)
# %%
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV

# Get the model and the predictions in (a) - (b)
def LGBM(q, X_train, Y_train, X_valid, Y_valid):
    params = {'n_estimators':[1000, 3000, 5000, 8000, 10000]}

    # (a) Modeling  
    model = LGBMRegressor(objective='quantile', alpha=q,
                         n_estimators=10000, bagging_fraction=0.7, learning_rate=0.027, subsample=0.7)                   
                         
    
    # grid_model = GridSearchCV(model, param_grid=params, scoring='neg_mean_squared_error', cv=5)
                         
    model.fit(X_train, Y_train, eval_metric = ['quantile'], 
          eval_set=[(X_valid, Y_valid)], early_stopping_rounds=300, verbose=500)

    # (b) Predictions
    
    return model
# %%
# Target 예측

def train_data(X_train, Y_train, X_valid, Y_valid, X_test, season):

    LGBM_models=[]
    LGBM_actual_pred = pd.DataFrame()

    for q in quantiles:
        models = []
        preds = pd.Series([-1]*season.size)
        for i, v in enumerate(['winter', 'fall', 'summer']):
            print(q, v)
            model = LGBM(q, X_train[i], Y_train[i], X_valid[i], Y_valid[i])
            pred = pd.Series(model.predict(X_test).round(2))[season == v]
            preds = preds.where(preds >= 0, pred)
            models.append(model)
        LGBM_models.append(models)
        LGBM_actual_pred = pd.concat([LGBM_actual_pred, preds],axis=1)

    LGBM_actual_pred.columns=quantiles
    
    return LGBM_models, LGBM_actual_pred
# %%

X_train_1 = (Xw_train_1, Xf_train_1, Xs_train_1)
Y_train_1 = (Yw_train_1, Yf_train_1, Ys_train_1)
X_valid_1 = (Xw_valid_1, Xf_valid_1, Xs_valid_1)
Y_valid_1 = (Yw_valid_1, Yf_valid_1, Ys_valid_1)
X_train_2 = (Xw_train_2, Xf_train_2, Xs_train_2)
Y_train_2 = (Yw_train_2, Yf_train_2, Ys_train_2)
X_valid_2 = (Xw_valid_2, Xf_valid_2, Xs_valid_2)
Y_valid_2 = (Yw_valid_2, Yf_valid_2, Ys_valid_2)
# Target1
models_1, results_1 = train_data(X_train_1, Y_train_1, X_valid_1, Y_valid_1, X_test, season)
results_1.sort_index()[:48]
# Target2
models_2, results_2 = train_data(X_train_2, Y_train_2, X_valid_2, Y_valid_2, X_test, season)
results_2.sort_index()[:48]

# results_2.sort_index()
# %%
submission = pd.read_csv('./data/sample_submission.csv')
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values
submission.to_csv('./submission/submission.csv', index=False)
# %%