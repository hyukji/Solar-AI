# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from module.data import get_train, get_test, save_submission

mpl.rcParams['figure.figsize'] = 6.4, 4.8

cons = 4
unit = 48
removed_cols = ['Hour', 'Minute']

df_train_x, df_train_y = get_train(cons, unit, removed_cols)
test_df = get_test(cons, unit, removed_cols)
# %%
def get_year(x, y): # year 1, 2, 3
    # 363일 365일 362일
    idx = [0, 17424, 34944, 52320]
    sidx = idx[:-1]
    eidx = idx[1:]
    for s, e in zip(sidx, eidx):
        yield x[s:e], y[s:e]

def get_month(x, y, month):
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cum = [0]
    for i in range(len(days)):
        cum.append(cum[i]+days[i])
    for i in range(len(cum)):
        cum[i] = 48*cum[i]
    sidx = cum[:-1]
    eidx = cum[1:]
    for x_year, y_year in get_year(x, y):
        for i, (s, e) in enumerate(zip(sidx, eidx)):
            if i + 1 == month:
                yield x_year[s:e], y_year[s:e]

data = list(get_month(df_train_x, df_train_y, 1))
len(data)
#%%
cnt = 0
for month in range(1, 13):
    # fig = plt.figure(figsize=(10,10))
    for x_train, y_train in get_month(df_train_x, df_train_y, month):
        cnt += 1
        # cols = ['TARGET_0', 'TARGET_1']
        cols = ['DHI_1', 'TARGET_1']
        # cols = ['T_0', 'WS_0']
        # ax = fig.add_subplot(1,1,1)
        plt.scatter(x_train[cols[0]], x_train[cols[1]])
        # for col in cols:
            # plt.plot(range(x_train['Day_0'].shape[0]), x_train[col])
        plt.xlabel(cols[0])
        plt.ylabel(cols[1])
        plt.xlim((0, 100))
        plt.ylim((0, 100))
        plt.title(f'{month}', loc='center')
        plt.legend([1, 2, 3])
    plt.show()
    # print(cnt)
    cnt=0
# plt.plot(df_train_x['Day_0'], df_train_x['TARGET_1'])
# %%
for x, y in zip(df_train_x, df_train_y):
    for i in range(cons):
        # f'TARGET_{i}'
df_train_y