import pandas as pd
import numpy as np


def merge_submission(file1, file2, merged_file):
    df1 = pd.read_csv(f"./test_data/{file1}.csv")
    df2 = pd.read_csv(f"./test_data/{file2}.csv")

    df1.iloc[:,1:] += df2.iloc[:,1:]
    df1.iloc[:,1:] /= 2

    df1.to_csv(f"./test_data/{merged_file}.csv", index=False)
    print("success to merge and saved!")



def LSTM_data(data, target, start_idx, end_idx, history_size, target_size, single_step = False):
    x_train = []
    y_train = []

    start_idx = start_idx + history_size
    end_idx = len(data) - target_size
    step = 48

    for i in range(start_idx, end_idx, step):
        indices = range(i - history_size, i)
        x_train.append(data[indices])

        if single_step:
            y_train.append(target[i+target_size])
        else:
            y_train.append(target[i:i+target_size])

    return np.array(x_train), np.array(y_train)




def get_train_data(cols, history_date, target_date):
    train_df = pd.read_csv("./data/train/train.csv")
    data_df = train_df[cols]
    data = data_df.values

    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    data = (data - data_mean) / data_std

    data = np.c_[data, train_df["TARGET"]]
    # data = np.where() # target ==0인값 지울까 고민중.

    history_size = history_date * 48 # 6일 * 48데이터
    target_size = target_date * 48 # 2일 * 48데이터
    x_train, y_train = LSTM_data(data, data[:,-1], 0, data.shape[0], history_size, target_size, single_step = False)
    

    return x_train, y_train


    

def LSTM_total_data(data, target, start_idx, end_idx, history_size, target_size, single_step = False):
    x_train = []
    y_train = []

    start_idx = start_idx + history_size
    end_idx = len(data) - target_size
    step = 48

    for i in range(start_idx, end_idx, step):
        indices = range(i - history_size, i)
        x_train.append(data[indices])

        if single_step:
            y_train.append(target[i+target_size])
        else:
            y_train.append(target[i:i+target_size])

    return np.array(x_train), np.array(y_train)




def get_train_total_data(cols, history_date, target_date):
    train_df = pd.read_csv("./data/train/train.csv")
    data_df = train_df[cols]
    data = data_df.values

    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    data = (data - data_mean) / data_std

    data = np.c_[data, train_df["TARGET"]]
    # data = np.where() # target ==0인값 지울까 고민중.
    
    history_size = history_date * 48 # 6일 * 48데이터
    target_size = target_date * 48 # 2일 * 48데이터
    x_train, y_train = LSTM_total_data(data, data[:,-1], 0, data.shape[0], history_size, target_size, single_step = False)
    

    return x_train, y_train

