import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import Input
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def tilted_loss(q,y,f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def NN(q, dims, X_train, Y_train, X_valid, Y_valid):

    # make sequential model
    model = Sequential()
    model.add(Input(shape=(dims[0],)))
    for dim in (dims[1:]):
        model.add(Dense(units=dim, activation='relu'))
    model.add(Dense(units=1, activation='relu')) # relu means removing minus value

    # object for model
    adam= Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    es= EarlyStopping(monitor='val_loss', mode='min', patience=10)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

    model.compile(loss=lambda y,f: tilted_loss(q,y,f), optimizer=adam)
    model.fit(X_train, Y_train, epochs=100, batch_size=256, verbose=2,
            validation_data=(X_valid, Y_valid), callbacks=[es, mc])
    
    return model


def train_data(dims, X_train, Y_train, X_valid, Y_valid):
    lst_models=[]
    for q in quantiles:
        print(q)
        model = NN(q, dims, X_train, Y_train, X_valid, Y_valid)
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
