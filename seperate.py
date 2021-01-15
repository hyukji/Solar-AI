# %%
import pandas as pd
import numpy as np

from module.data import get_train, get_test, save_submission, concat_data
# %%
cons = 96
unit = 1
removed_cols = ['Day', 'Minute', 'Hour']

df_train_x, df_train_y = get_train(cons, unit, removed_cols)
X_test = get_test(cons, unit, removed_cols)
df_train_x.shape

#%%
df_train_x = df_train_x[df_train_x.index % 48 == 0]
index = ['sp', 'su', 'f', 'w']

train_labels = [3]*48*59 + [0]*48*91 +[1]*48*92 + [2]*48*91 + [3]*48*31
train_labels *= 3
train_labels += [3]*48 # 마지막에 겨울 라벨 추가

train_labels = [3]*59 + [0]*91 +[1]*92 + [2]*91 + [3]*31
train_labels *= 3
# train_labels = [3]*29 + [0]*46 +[1]*46 + [2]*45 + [3]*16
# train_labels *= 3
# train_labels += [3]
train_labels = pd.Series(train_labels)
# %%
# Normalize
# mean = df_train_x.mean(axis=0)
# std = df_train_x.std(axis=0)
# df_train_x = (df_train_x - mean) / std
# X_test = (X_test - mean) / std

from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split(df_train_x, train_labels, test_size=0.2, random_state=0)
# %%

# %%
# get absolute test data from train data (For last cell)
X_train, ttx, Y_train, tty = train_test_split(X_train, Y_train, test_size=0.1, random_state=1)
# %%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def NN(dims, params, X_train, Y_train, X_valid, Y_valid):

    # make sequential model
    model = Sequential()
    model.add(Input(shape=(dims[0],)))
    for dim in (dims[1:]):
        model.add(Dense(units=dim, activation='relu'))
        model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units=4, activation='softmax'))

    # object for model
    adam= Adam(lr=params['lr'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    es= EarlyStopping(monitor='val_loss', mode='min', patience=3)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=2,
            validation_data=(X_valid, Y_valid), callbacks=[es, mc])
    
    return model

dims = [576, 512, 128]
# dims = [192, 512, 128] , 0.05
params = {'lr': 0.005}
# Target1
model = NN(dims, params, X_train, Y_train, X_valid, Y_valid)

# %%
model.evaluate(ttx, tty)

X_test = X_test[X_test.index % 48 == 0]
pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)

# 학습에 맞는 사이즈로 복원
test_labels = pd.Series(index=range(X_test.shape[0]*48))
for i in range(pred.size): # 81개
    test_labels.iloc[i*48] = pred[i]
test_labels = test_labels.fillna(method='ffill')
# %%

cons = 4
unit = 48
removed_cols = ['Day', 'Minute']

df_train_x, df_train_y = get_train(cons, unit, removed_cols)

train_labels = [3]*48*59 + [0]*48*91 +[1]*48*92 + [2]*48*91 + [3]*48*31
train_labels *= 3
# train_labels += [3]*48 # 마지막에 겨울 라벨 추가
train_labels = train_labels[96:]
train_labels = pd.Series(train_labels)
print(df_train_x.shape, train_labels.size)
df_train_x['season'] = train_labels

df_train = pd.concat([df_train_x, df_train_y], axis=1)

sp_train = df_train[df_train.season == 0]
su_train = df_train[df_train.season == 1]
f_train = df_train[df_train.season == 2]
w_train = df_train[df_train.season == 3]

X_test = get_test(cons, unit, removed_cols)
X_test['season'] = test_labels

# %%
Xsp_train_1, Ysp_train_1 = sp_train.iloc[:, :-2], sp_train.iloc[:, -2]
Xsp_train_2, Ysp_train_2 = sp_train.iloc[:, :-2], sp_train.iloc[:, -1]

Xsu_train_1, Ysu_train_1 = su_train.iloc[:, :-2], su_train.iloc[:, -2]
Xsu_train_2, Ysu_train_2 = su_train.iloc[:, :-2], su_train.iloc[:, -1]

Xf_train_1, Yf_train_1 = f_train.iloc[:, :-2], f_train.iloc[:, -2]
Xf_train_2, Yf_train_2 = f_train.iloc[:, :-2], f_train.iloc[:, -1]

Xw_train_1, Yw_train_1 = w_train.iloc[:, :-2], w_train.iloc[:, -2]
Xw_train_2, Yw_train_2 = w_train.iloc[:, :-2], w_train.iloc[:, -1]

# Xw_train_1, Xw_valid_1, Yw_train_1, Yw_valid_1 = train_test_split(w_train.iloc[:, :-2], w_train.iloc[:, -2], test_size=0.3, random_state=0)
# Xw_train_2, Xw_valid_2, Yw_train_2, Yw_valid_2 = train_test_split(w_train.iloc[:, :-2], w_train.iloc[:, -1], test_size=0.3, random_state=0)

# Xf_train_1, Xf_valid_1, Yf_train_1, Yf_valid_1 = train_test_split(f_train.iloc[:, :-2], f_train.iloc[:, -2], test_size=0.3, random_state=0)
# Xf_train_2, Xf_valid_2, Yf_train_2, Yf_valid_2 = train_test_split(f_train.iloc[:, :-2], f_train.iloc[:, -1], test_size=0.3, random_state=0)

# Xsp_train_1, Xsp_valid_1, Ysp_train_1, Ysp_valid_1 = train_test_split(sp_train.iloc[:, :-2], sp_train.iloc[:, -2], test_size=0.3, random_state=0)
# Xsp_train_2, Xsp_valid_2, Ysp_train_2, Ysp_valid_2 = train_test_split(sp_train.iloc[:, :-2], sp_train.iloc[:, -1], test_size=0.3, random_state=0)

# Xsu_train_1, Xsu_valid_1, Ysu_train_1, Ysu_valid_1 = train_test_split(su_train.iloc[:, :-2], su_train.iloc[:, -2], test_size=0.3, random_state=0)
# Xsu_train_2, Xsu_valid_2, Ysu_train_2, Ysu_valid_2 = train_test_split(su_train.iloc[:, :-2], su_train.iloc[:, -1], test_size=0.3, random_state=0)

# print(Xw_train_1.shape, Xf_train_1.shape, Xsp_train_1.shape, Xsu_train_1.shape)
# print(Xw_valid_1.shape, Xf_valid_1.shape, Xsp_valid_1.shape, Xsu_valid_1.shape)
# %%
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

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

def train_data(Xtrain, Ytrain):

    kfold = KFold(n_splits=2, shuffle=True, random_state=0)

    LGBM_models=[]

    for q in quantiles:
        models = []
        for i in range(4):
            models_season = []
            fold = 0
            for train_index, test_index in kfold.split(Xtrain[i],Ytrain[i]):
                print(f'quantile: {q}, season: {i}')
                print(f'{fold}th fold')
                X_train, X_valid = Xtrain[i].iloc[train_index,:], Xtrain[i].iloc[test_index, :]
                Y_train, Y_valid = Ytrain[i].iloc[train_index], Ytrain[i].iloc[test_index]
                model = LGBM(q, X_train, Y_train, X_valid, Y_valid)
                models_season.append(model)
                fold += 1
            models.append(models_season)
        LGBM_models.append(models)
    
    return LGBM_models

X_train_1 = (Xsp_train_1, Xsu_train_1, Xf_train_1, Xw_train_1)
Y_train_1 = (Ysp_train_1, Ysu_train_1, Yf_train_1, Yw_train_1)
# X_valid_1 = (Xsp_valid_1, Xsu_valid_1, Xf_valid_1, Xw_valid_1)
# Y_valid_1 = (Ysp_valid_1, Ysu_valid_1, Yf_valid_1, Yw_valid_1)

X_train_2 = (Xsp_train_2, Xsu_train_2, Xf_train_2, Xw_train_2)
Y_train_2 = (Ysp_train_2, Ysu_train_2, Yf_train_2, Yw_train_2)
# X_valid_2 = (Xsp_valid_2, Xsu_valid_2, Xf_valid_2, Xw_valid_2)
# Y_valid_2 = (Ysp_valid_2, Ysu_valid_2, Yf_valid_2, Yw_valid_2)
# %%
# preds = pd.Series([np.nan]*X_test.shape[0])
# pred = pd.Series(m.predict(X_test).round(2))[(X_test.season == 0).reset_index(drop=True)]
# print(pred)
# preds = preds.where(preds.notnull(), pred)
# preds[188]
# %%
models_1 = train_data(X_train_1, Y_train_1)
# Target2
models_2 = train_data(X_train_2, Y_train_2)
# %%
def predict(models):
    LGBM_actual_pred = pd.DataFrame()
    for j, qu in enumerate(models):
        preds = pd.Series([np.nan]*X_test.shape[0])
        for i, ms in enumerate(qu):
            print(f'quanti {j+1}, season {i}')
            if i in [0, 1]: # spring or summer:
                continue
            pred_aver = pd.Series([0]*X_test.shape[0])
            for m in ms:
                pred = pd.Series(m.predict(X_test).round(2))[(X_test.season == i).reset_index(drop=True)]
                pred_aver += pred
            pred_aver /= len(ms) # fold num
            preds = preds.where(preds.notnull(), pred_aver)
        # print(preds.isnull().values.any())
        LGBM_actual_pred = pd.concat([LGBM_actual_pred, preds],axis=1)
    LGBM_actual_pred.columns=quantiles
    return LGBM_actual_pred

results_1 = predict(models_1)
results_2 = predict(models_2)
# %%
print(results_1.isnull().values.any())
# %%
r1 = pd.read_pickle('./results_1.pkl')
r2 = pd.read_pickle('./results_2.pkl')

results_1 = results_1.where(results_1.notnull(), r1)
results_2 = results_1.where(results_2.notnull(), r1)

# %%
submission = pd.read_csv('./data/sample_submission.csv')
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values
submission.to_csv('./submission/sepa_lgbm.csv', index=False)
# %%
