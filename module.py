import pandas as pd
import numpy as np
import random

import warnings
warnings.filterwarnings("ignore")

def add_future_feats(data, after_days, col):
    # day: 1~
    temp = data.copy()
    new_cols = []
    for i, day in enumerate(after_days):
        new_col = f'{col}_after_{i+1}'
        temp[new_col] = temp[col].shift(-(day*48), fill_value = np.nan)
        new_cols.append(new_col)
    return temp

def concat_data(data, consecutive, unit): # num of consecutive days
    temp = data.copy()
    res = data.copy()
    cols_name = res.columns
    res.columns = [f'{col}_{0}' for col in cols_name]
    for ele in range(1, consecutive):
        new_ele = temp.shift(-ele*unit, fill_value = np.nan)
        new_ele.columns = [f'{col}_{ele}' for col in cols_name]
        res = pd.concat([res, new_ele],axis=1)
    return res

# for test
def get_one_data(data): # day: 0~6
    # assert data.shape[0] == 336 # 48*7
    # temp = data.copy()
    # res = pd.DataFrame()
    return data.iloc[:48]
    # for idx, day in enumerate(range(6, 6-n_days, -1)):
    #     one_day = temp.iloc[day*48:(day+1)*48]
    #     one_day.reset_index(drop=True, inplace=True)
    #     cols_name = one_day.columns
    #     one_day.columns = [f'{col}_{idx}' for col in cols_name]
    #     res = pd.concat([res, one_day],axis=1)
    # return res


def preprocess_data(data, consecutive, unit, is_train=True):
    # 원하는 칼럼 추가는 여기서
    # ex, temp['GHI'] = temp['DHI'] + temp['DNI']
    temp = data.copy()
    removed_cols = ['Day', 'Minute']

    temp = temp.drop(removed_cols, axis='columns')
    temp = concat_data(temp, consecutive, unit)
    if is_train:
        after_days = [1, 2]
        col = f'TARGET_{consecutive-1}'

        temp = add_future_feats(temp, after_days, col)
        temp = temp.dropna()
        return temp

    else:
        temp = get_one_data(temp)
        return temp


def get_train(consecutive, unit, save=False):
    temp = pd.read_csv('./data/train/train.csv')
    df_train = preprocess_data(temp, consecutive, unit, is_train=True)
    if save:
        df_train.to_pickle('./df_train.pkl')
    return df_train


def get_test(consecutive, unit, save=False):
    df_test = []

    for i in range(81):
        file_path = './data/test/' + str(i) + '.csv'
        temp = pd.read_csv(file_path)
        temp = preprocess_data(temp, consecutive, unit, is_train=False)
        df_test.append(temp)

    X_test = pd.concat(df_test)
    if save:
        X_test.to_pickle('./X_test.pkl')
    return X_test