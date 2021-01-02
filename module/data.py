import pandas as pd
import numpy as np
import random

import warnings
warnings.filterwarnings("ignore")

# col을 기준으로 after_days만큼 지난 column을 data 끝에 붙여서 반환함
def add_future_feats(data, after_days, col):
    # after_days : list of day
    # col : standard of shifting
    temp = data.copy()
    new_cols = []
    for i, day in enumerate(after_days):
        new_col = f'{col}_after_{i+1}'
        temp[new_col] = temp[col].shift(-(day*48), fill_value = np.nan)
        new_cols.append(new_col)
    return temp

# 간격이 unit인 consecutive개의 데이터를 이어붙인 df를 반환함
def concat_data(data, consecutive, unit):
    # consecutive : num of consecutive days
    # unit : interval (1 day = 48)
    temp = data.copy()
    res = data.copy()
    cols_name = res.columns
    res.columns = [f'{col}_{0}' for col in cols_name] # 시작점은 _0으로 이름 변환
    for ele in range(1, consecutive):
        new_ele = temp.shift(-ele*unit, fill_value = np.nan)
        new_ele.columns = [f'{col}_{ele}' for col in cols_name] # 추가되는 df의 column마다 _n 을 붙임
        res = pd.concat([res, new_ele],axis=1)
    return res

# test의 마지막 48개의 행을 df로 반환함 (나중의 데이터를 우선시)
def get_one_data(data):
    return data.iloc[-48:]
    # assert data.shape[0] == 336 # 48*7
    # temp = data.copy()
    # res = pd.DataFrame()
    # for idx, day in enumerate(range(6, 6-n_days, -1)):
    #     one_day = temp.iloc[day*48:(day+1)*48]
    #     one_day.reset_index(drop=True, inplace=True)
    #     cols_name = one_day.columns
    #     one_day.columns = [f'{col}_{idx}' for col in cols_name]
    #     res = pd.concat([res, one_day],axis=1)
    # return res

# train과 test의 데이터 전처리
def preprocess_data(data, consecutive, unit, removed_cols = ['Day', 'Hour', 'Minute'], is_train=True):
    # 원하는 칼럼 추가 및 삭제 가능
    # ex, temp['GHI'] = temp['DHI'] + temp['DNI']

    # train, test 공통 작업
    temp = data.copy()
    temp = temp.drop(removed_cols, axis='columns')
    temp = concat_data(temp, consecutive, unit)
    
    if is_train:
        after_days = [1, 2]
        col = f'TARGET_{consecutive-1}' # concat_data에서 마지막으로 추가된 df의 target column 이름

        temp = add_future_feats(temp, after_days, col) # Target 데이터 추가
        temp = temp.dropna()
        return temp

    else:
        temp = temp.dropna()
        temp = get_one_data(temp)
        return temp

# train 파일 입출력 및 전처리
def get_train(consecutive, unit, save=False):
    temp = pd.read_csv('./data/train/train.csv')
    df_train = preprocess_data(temp, consecutive, unit, is_train=True)
    if save:
        df_train.to_pickle('./df_train.pkl')
    
    # X, Y를 분리하여 반환
    return df_train.iloc[:, :-2], df_train.iloc[:, -2:]

# test 파일 입출력 및 전처리
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

# sort_index isn't necessary
def save_submission(results_1, results_2, file_name):
    submission = pd.read_csv('./data/sample_submission.csv')
    submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
    submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values
    submission.to_csv(f'./submission/{file_name}.csv', index=False)