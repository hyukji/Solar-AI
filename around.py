# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from module.data import get_train, get_test, save_submission
from module.data import concat_data

removed_cols = ['Day', 'Minute']

df_train_x, df_train_y = get_train(4, 48, removed_cols)
df_train_x = concat_data(df_train_x, 2, -1)
df_train_x = df_train_x.fillna(method="bfill")

X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(df_train_x, df_train_y.iloc[:, 0], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(df_train_x, df_train_y.iloc[:, 1], test_size=0.3, random_state=0)

X_test = get_test(4, 48, removed_cols)
# %%
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# quantiles = [0.4, 0.5, 0.6]
from lightgbm import LGBMRegressor

# Get the model
def LGBM(q, X_train, Y_train, X_valid, Y_valid):
    model = LGBMRegressor(objective='quantile', alpha=q,
                         n_estimators=10000, bagging_fraction=0.7, learning_rate=0.027, subsample=0.7)
    model.fit(X_train, Y_train, eval_metric = ['quantile'], 
            eval_set=[(X_valid, Y_valid)], early_stopping_rounds=300, verbose=500)
    return model


def train_data(X_train, Y_train, X_valid, Y_valid):
    lst_models=[]
    for q in quantiles:
        print(q)    
        model = LGBM(q, X_train, Y_train, X_valid, Y_valid)
        lst_models.append(model)
    
    return lst_models

def test_data(models, X_test):
    df_pred = pd.DataFrame()
    for model in models:
        pred = pd.Series(model.predict(X_test).round(2))
        df_pred = pd.concat([df_pred, pred],axis=1)

    df_pred.columns=quantiles
    return df_pred

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
from  lightgbm import plot_importance
for model in models_1:
    plot_importance(model)
# %%
submission = pd.read_csv('./data/sample_submission.csv')
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values
submission.to_csv('./submission/around.csv', index=False)
# %%
# x1test, y1test의 순서가 같아야 함을 유의
import tensorflow.keras.backend as K
def tilted_loss(q,y,f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

for q, model in zip(quantiles, models_1):
    pred = model.predict(x1test).round(2)
    print(tilted_loss(q, y1test, pred))

for q, model in zip(quantiles, models_2):
    pred = model.predict(x2test).round(2)
    print(tilted_loss(q, y2test, pred))
# %%
from  lightgbm import plot_importance
for model in models_1:
    plot_importance(model)
# %%
