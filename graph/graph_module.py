#%%
import pandas as pd
import numpy as np
import random

import warnings
warnings.filterwarnings("ignore")


def get_train_graph():
    train_df = pd.read_csv("../data/train/train.csv")
    season_df = pd.read_csv("../data/train_season.csv",  header = None, index_col = 0, squeeze = True)
    
    for d in range(1095):
        indices = range(48 * d, 48 * (d + 1))
        season = 3 if d <= 3 else season_df[d - 3]

        train_df.loc[indices, "SEASON"] = season

    return train_df


train_df = get_train_graph()


# %%
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)
season_arr = ["Spring", "Summer","Fall","Winter"]
for s in range(4):
    season_df = train_df[train_df["SEASON"] == s]
    
    df = season_df.reset_index()
    
    df = df["TARGET"]
    
    plt.title(season_arr[s])
    df.plot()
    plt.savefig(f'index-target_{season_arr[s]}.png')
    plt.show()


# %%
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)

df = train_df.loc[0: 17000, "TARGET"]
# plt.title(season_arr[s])
df.plot()


df = train_df.loc[17000: 34000, "TARGET"]
# plt.title(season_arr[s])
df.plot()


# %%

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)
season_arr = ["Spring", "Summer","Fall","Winter"]
for s in range(4):
    season_df = train_df[train_df["SEASON"] == s]
    # train_df["Hour"] = train_df.where(train_df["Hour"] == 0, train_df["Hour"] , train_df["Hour"]  + 0.5)
    # train_df.set_index("Day")
    df = season_df.reset_index()
    scope = int(len(season_df) / 3)
    df = df.loc[: scope, :]
    
    df = df[["DNI", "DHI"]]

    plt.title(season_arr[s])
    df.plot()
    plt.savefig(f'DNI-DHI{season_arr[s]}.png')



# %%

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)
season_arr = ["Spring", "Summer","Fall","Winter"]
for s in range(4):
    season_df = train_df[train_df["SEASON"] == s]

    
    scope = int(len(season_df) / 3)
    df = df.loc[:scope, :]
    
    df = season_df.reset_index()
    df["DHI/DNI"] = df["DHI"] / (df["DNI"] + df["DHI"])
    
    df = df[["TARGET", "T"]]
    
    df.plot()
    plt.title(season_arr[s])

    
    # WS - 풍속(Wind Speed (m/s))
    # RH - 상대습도(Relative Humidity (%))
    # T - 기온(Temperature (Degree C))
