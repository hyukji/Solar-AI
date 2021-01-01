# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import tensorflow.keras.backend as K
import tensorflow as tf
# %%
df_train = pd.read_pickle('./df_train.pkl')
X_test = pd.read_pickle('./X_test.pkl')

xx = df_train.iloc[:, :-2]

mean = xx.mean(axis=0)
std = xx.std(axis=0)
xx = (xx - mean) / std
X_test = (X_test - mean) / std

from sklearn.model_selection import train_test_split
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(xx, df_train.iloc[:, -2], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(xx, df_train.iloc[:, -1], test_size=0.3, random_state=0)
# %%
def tilted_loss(q,y,f):
    e = (y-f)
    print(e)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)


def mcycleModel():
    model = Sequential()
    model.add(tf.keras.Input(shape=(768,)))
    model.add(Dense(units=284,activation='relu'))
    model.add(Dense(units=80,activation='relu'))
    # model.add(Dense(units=512,activation='relu'))
    model.add(Dense(1))
    
    return model

qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# qs = [0.5]
# q1 0.2 - lr 0.001 - loss 1.0
# q2 0.2 - lr 0.001 - loss 1.1

def train_data(X_train, Y_train, X_valid, Y_valid, X_test):

#quantile 0.2랑 lr 0.005랑 안 맞음
# 0.9 3.5 2.? 2.? 1.8 1.68 ? ? 0.66
# 0.9 3.5 1.8 2.0 8.8 ? 1.5 1.15 0.66
    models=[]
    actual_pred = pd.DataFrame()
    for q in qs:
        print(q)
        model = mcycleModel()
        adam= tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        es= tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10)
        mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

        model.compile(loss=lambda y,f: tilted_loss(q,y,f), optimizer=adam)
        model.fit(X_train, Y_train, epochs=50, batch_size=256, verbose=2, validation_data=(X_valid, Y_valid), callbacks=[es, mc])
        
        # Predict the quantile
        y_test = model.predict(X_test)
        y_test = np.squeeze(y_test)
        pred = pd.Series(y_test.round(2))

        models.append(model)
        actual_pred = pd.concat([actual_pred,pred],axis=1)
    actual_pred = pd.concat([actual_pred,pred],axis=1)
    return models, actual_pred
# %%
# Target1
models_1, results_1 = train_data(X_train_1, Y_train_1, X_valid_1, Y_valid_1, X_test)
results_1.sort_index()[:48]

# %%
# Target2
models_2, results_2 = train_data(X_train_2, Y_train_2, X_valid_2, Y_valid_2, X_test)
results_2.sort_index()[:48]

# %%
actual = pd.DataFrame()
for q, model in zip(qs, models_1):
    pred = model.predict(xxx).round(2)
    pred = np.squeeze(pred)
    pred = pd.Series(pred.round(2))
    print(tilted_loss(q, yyy, pred))

#%%
print(Y_train_1.sort_index() - pred)
# %%
submission = pd.read_csv('./data/sample_submission.csv')
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values
submission.to_csv('./submission/submission.csv', index=False)
# %%
results_1.to_pickle('results_1.pkl')
results_2.to_pickle('results_2.pkl')
r1=results_1.where(results_1 >=  0 , 0)
r2=results_2.where(results_2 >=0, 0)

submission = pd.read_csv('./data/sample_submission.csv')
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = r1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = r2.sort_index().values
submission.to_csv('./submission/submission_revised.csv', index=False)
# %%
