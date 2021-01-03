
# %%
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K
# %%
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def tilted_loss(q,y,f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def RNN(q, dims, params, X_train, Y_train, X_valid, Y_valid):

    timesteps = 4
    input_dim = 6 # day만 제외
    hidden_size = 20

    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(timesteps, input_dim), return_sequences = True))
    # model.add(LSTM(hidden_size, return_sequences = True))
    model.add(Dense(units=1, activation='relu'))

    # object for model
    adam= Adam(lr=params['lr'])
    es= EarlyStopping(monitor='val_loss', mode='min', patience=10)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

    model.compile(loss=lambda y,f: tilted_loss(q,y,f), optimizer=adam)
    model.fit(X_train, Y_train, epochs=100, batch_size=256, verbose=2,
            validation_data=(X_valid, Y_valid), callbacks=[es, mc])
    
    return model

def train_data(dims, params, X_train, Y_train, X_valid, Y_valid):
    lst_models=[]
    for q in quantiles:
        print(q)
        model = RNN(q, dims, params, X_train, Y_train, X_valid, Y_valid)
        lst_models.append(model)

    return lst_models

def test_data(models, X_test):
    df_pred = pd.DataFrame()
    for model in models:
        pred = model.predict(X_test)
        pred = np.squeeze(pred)
        pred = pd.Series(pred.round(2))
        df_pred = pd.concat([df_pred, pred],axis=1)

    df_pred.columns=quantiles
    return df_pred

# %%

import pandas as pd
import numpy as np

from module.data import get_train, get_test, save_submission

cons = 4
unit = 48
removed_cols = ['Day', 'Hour', 'Minute']

df_train_x, df_train_y = get_train(cons, unit, removed_cols)
x_train = df_train_x.to_numpy()
x_train = np.array(np.hsplit(x_train, cons)).transpose(1, 0, 2)
y_train = df_train_y.to_numpy()
# x_train.shape = (num data, timesteps(=cons), num features)

X_test = get_test(cons, unit, removed_cols)
X_test = X_test.to_numpy()
X_test = np.array(np.hsplit(X_test, cons)).transpose(1, 0, 2)

from sklearn.model_selection import train_test_split
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(x_train, y_train[:, 0], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(x_train, y_train[:, 1], test_size=0.3, random_state=0)

# %%
dims = [384, 180, 60] # 안 씀 hidden, time, input_dim 바꾸는 데에 쓰든지
params = {'lr': 0.005}
# Target1
models_1 = train_data(dims, params, X_train_1, Y_train_1, X_valid_1, Y_valid_1)
results_1 = test_data(models_1, X_test)
results_1[:48]

# Target2
models_2 = train_data(dims, params, X_train_2, Y_train_2, X_valid_2, Y_valid_2)
results_2 = test_data(models_2, X_test)
results_2[:48]
# %%
# from tensorflow.keras.datasets import reuters
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical

# (X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=1000, test_split=0.2)
# max_len = 100
# X_train = pad_sequences(X_train, maxlen=max_len) # 훈련용 뉴스 기사 패딩
# X_test = pad_sequences(X_test, maxlen=max_len) # 테스트용 뉴스 기사 패딩
# # %%
# X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
# X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))
# %%
