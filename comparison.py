
#%%
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

from scipy.stats import norm
from sklearn import ensemble

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow import keras

sns.set_style('white')
DPI = 200
mpl.rc('savefig', dpi=DPI)
mpl.rcParams['figure.dpi'] = DPI
mpl.rcParams['figure.figsize'] = 6.4, 4.8  # Default.
mpl.rcParams['font.sans-serif'] = 'Roboto'
mpl.rcParams['font.family'] = 'sans-serif'

# Set title text color to dark gray (https://material.io/color) not black.
TITLE_COLOR = '#212121'
mpl.rcParams['text.color'] = TITLE_COLOR

# Axis titles and tick marks are medium gray.
AXIS_COLOR = '#757575'
mpl.rcParams['axes.labelcolor'] = AXIS_COLOR
mpl.rcParams['xtick.color'] = AXIS_COLOR
mpl.rcParams['ytick.color'] = AXIS_COLOR
# %%
from module.data import get_train, get_test, save_submission

cons = 2
unit = 48
removed_cols = ['Day', 'Hour', 'Minute']

df_train_x, df_train_y = get_train(cons, unit, removed_cols)
test_df = get_test(cons, unit, removed_cols)

from sklearn.model_selection import train_test_split
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(df_train_x, df_train_y.iloc[:, 0], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(df_train_x, df_train_y.iloc[:, 1], test_size=0.3, random_state=0)

train_df = X_train_2
test_df = X_valid_2
train_labels = Y_train_2
test_labels = Y_valid_2
# %%
train_df.describe()
# %%
# Normalize
mean = train_df.mean(axis=0)
std = train_df.std(axis=0)
train_df = (train_df - mean) / std
test_df = (test_df - mean) / std

# x_train = train_df.DNI_1
# x_test = test_df.DNI_1

# max idx = cons - 1
def denorm(df, idx):
    x = df.iloc[:, idx]
    return x * std[idx] + mean[idx]

# x_train_denorm = denorm(x_train)
# x_test_denorm = denorm(x_test)

X_train = sm.add_constant(train_df.copy(deep=True))
X_test = sm.add_constant(test_df.copy(deep=True))
# %%
METHODS = ['OLS', 'QuantReg', 'Random forests', 'Gradient boosting']
            #, 'Keras', 'ensorFlow']

# QUANTILES = [0.1, 0.5, 0.9]
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# QUANTILES.reverse()  # Test out to see if we're getting different results.

quantiles_legend = [str(int(q * 100)) + 'th percentile' for q in QUANTILES]

# sns.set_palette(sns.color_palette('Blues', len(QUANTILES)))
sns.set_palette(sns.color_palette('Blues'))
# Set dots to a light gray
dot_color = sns.color_palette('coolwarm', 3)[1]
#%%
preds = np.array([(method, q, x) 
                  for method in METHODS 
                  for q in QUANTILES
                  for x in X_test.const])
preds = pd.DataFrame(preds)
preds.columns = ['method', 'q', 'x']
preds = preds.apply(lambda x: pd.to_numeric(x, errors='ignore'))

preds['label'] = np.resize(test_labels, preds.shape[0])
preds = preds[['method', 'q', 'label']]
# %%
# pandas version rather than Keras.
def quantile_loss(q, y, f):
    # q: Quantile to be evaluated, e.g., 0.5 for median.
    # y: True value.
    # f: Fitted or predicted value.
    e = y - f
    return np.maximum(q * e, (q - 1) * e)
# %%
from matplotlib.ticker import FuncFormatter

ax = plt.scatter(denorm(X_train, 1), train_labels, color=dot_color)
plt.title('DNI_1 vs. Target_1 (training slice)', loc='left')
sns.despine(left=True, bottom=True)
ax.axes.xaxis.set_major_formatter(FuncFormatter(
    lambda x, _: '{}'.format(x)))
ax.axes.yaxis.set_major_formatter(FuncFormatter(
    lambda y, _: '{}'.format(y)))
plt.xlabel('Proportion of owner-occupied units built prior to 1940')
plt.ylabel('Median value of owner-occupied homes')
plt.show()
# %%
ols = sm.OLS(train_labels, X_train).fit()

def ols_quantile(m, X, q):
    # m: OLS model.
    # X: X matrix.
    # q: Quantile.
    #
    # Set alpha based on q. Vectorized for different values of q.
    mean_pred = m.predict(X)
    se = np.sqrt(m.scale)
    return mean_pred + norm.ppf(q) * se
# preds.loc[preds.method == 'OLS', 'pred'] = np.concatenate(
#     [ols_quantile(ols, X_test, q) for q in QUANTILES]) 

# %%
quantreg = sm.QuantReg(train_labels, X_train)  # Don't fit yet, since we'll fit once per quantile.
# preds.loc[preds.method == 'QuantReg', 'pred'] = np.concatenate(
#     [quantreg.fit(q=q).predict(X_test) for q in QUANTILES]) 
# %%
N_ESTIMATORS = 1000
rf = ensemble.RandomForestRegressor(n_estimators=N_ESTIMATORS, 
                                    min_samples_leaf=1, random_state=3, 
                                    verbose=True, 
                                    n_jobs=-1)  # Use maximum number of cores.
rf.fit(X_train, train_labels)

def rf_quantile(m, X, q):
    rf_preds = []
    for estimator in m.estimators_:
        rf_preds.append(estimator.predict(X))
    rf_preds = np.array(rf_preds).transpose()  # One row per record.
    return np.percentile(rf_preds, q * 100, axis=1)

# preds.loc[preds.method == 'Random forests', 'pred'] = np.concatenate(
#     [rf_quantile(rf, X_test, q) for q in QUANTILES]) 
# %%
def gb_quantile(X_train, train_labels, X, q):
    gbf = ensemble.GradientBoostingRegressor(loss='quantile', alpha=q,
                                             n_estimators=N_ESTIMATORS,
                                             max_depth=3,
                                             learning_rate=0.1, min_samples_leaf=9,
                                             min_samples_split=9, verbose=1)
    gbf.fit(X_train, train_labels)
    return gbf.predict(X)

# preds.loc[preds.method == 'Gradient boosting', 'pred'] = np.concatenate(
#     [gb_quantile(X_train, train_labels, X_test, q) for q in QUANTILES])
# %%
# for keras and tensorflow
# x_train_expanded = np.expand_dims(x_train, 1)
# x_test_expanded = np.expand_dims(x_test, 1)
# train_labels_expanded = np.expand_dims(train_labels, 1)

# EPOCHS = 200
# BATCH_SIZE = 32
# UNITS = 512

# def tilted_loss(q, y, f):
#     e = (y - f)
#     return keras.backend.mean(keras.backend.maximum(q * e, (q - 1) * e), axis=-1)
# optimizer = tf.train.AdamOptimizer(0.001)
# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

# def keras_pred(x_train, train_labels, x_test, q):
#     print(q)
#     # Set input_dim for the number of features.
#     if len(x_train.shape) == 1:
#         input_dim = 1
#     else:
#         input_dim = x_train.shape[1]
#     model = keras.Sequential([
#       keras.layers.Dense(UNITS, activation=tf.nn.relu,
#                          input_dim=input_dim),
#       keras.layers.Dense(UNITS, activation=tf.nn.relu),
#       keras.layers.Dense(1)
#     ])
    
#     model.compile(loss=lambda y, f: tilted_loss(q, y, f), optimizer=optimizer)
#     model.fit(x_train, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE,
#               verbose=0, validation_split=0.2, callbacks=[early_stop])
    
#     # Predict the quantile
#     return model.predict(x_test)

# preds.loc[preds.method == 'Keras', 'pred'] = np.concatenate(
#     [keras_pred(x_train_expanded, train_labels, x_test_expanded, q) 
#      for q in QUANTILES]) 
# %%
for i, method in enumerate(METHODS):
    ax = plt.scatter(denorm(X_test, 0), test_labels, color=dot_color)
    plt.plot(preds[preds.method == method].pivot_table(
        index='x_denorm', columns='q', values='pred'))
    plt.legend(quantiles_legend)
    # Reversing legend isn't working, possibly because of multiple plots.
#     handles, labels = ax.get_legend_handles_labels()
#     ax.legend(handles[::-1], labels[::-1])
    plt.xlim((0, 100))
    ax.axes.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: '{}'.format(x)))
    ax.axes.yaxis.set_major_formatter(FuncFormatter(
        lambda y, _: '${}k'.format(y)))
    plt.xlabel('Proportion of owner-occupied units built prior to 1940')
    plt.ylabel('Median value of owner-occupied homes')
    plt.title(method + ' quantiles', loc='left')
    sns.despine(left=True, bottom=True)
    plt.show()
# %%
preds['quantile_loss'] = quantile_loss(preds.q, preds.label, preds.pred)

def plot_loss_comparison(preds):
    overall_loss_comparison = preds[~preds.quantile_loss.isnull()].\
      pivot_table(index='method', values='quantile_loss').\
      sort_values('quantile_loss')
    # Show overall table.
    print(overall_loss_comparison)
  
    # Plot overall.
    with sns.color_palette('Blues', 1):
        ax = overall_loss_comparison.plot.barh()
        plt.title('Total quantile loss', loc='left')
        sns.despine(left=True, bottom=True)
        plt.xlabel('Quantile loss')
        plt.ylabel('')
        ax.legend_.remove()
  
    # Per quantile.
    per_quantile_loss_comparison = preds[~preds.quantile_loss.isnull()].\
        pivot_table(index='q', columns='method', values='quantile_loss')
    # Sort by overall quantile loss.
    per_quantile_loss_comparison = \
        per_quantile_loss_comparison[overall_loss_comparison.index]
    print(per_quantile_loss_comparison)
  
    # Plot per quantile.
    with sns.color_palette('Blues'):
        ax = per_quantile_loss_comparison.plot.barh()
        plt.title('Quantile loss per quantile', loc='left')
        sns.despine(left=True, bottom=True)
        handles, labels = ax.get_legend_handles_labels()
        plt.xlabel('Quantile loss')
        plt.ylabel('Quantile')
        # Reverse legend.
        ax.legend(reversed(handles), reversed(labels))
plot_loss_comparison(preds)

# %%
ols_full = sm.OLS(train_labels, X_train).fit()
preds.loc[preds.method == 'OLS', 'pred'] = np.concatenate(
    [ols_quantile(ols_full, X_test, q) for q in QUANTILES]) 
# %%
# Don't fit yet, since we'll fit once per quantile.
quantreg_full = sm.QuantReg(train_labels, X_train)
preds.loc[preds.method == 'QuantReg', 'pred'] = np.concatenate(
    [quantreg_full.fit(q=q).predict(X_test) for q in QUANTILES])
# %%
rf_full = ensemble.RandomForestRegressor(n_estimators=N_ESTIMATORS, 
                                         min_samples_leaf=1, random_state=3, 
                                         n_jobs=-1)
rf_full.fit(X_train, train_labels)

preds.loc[preds.method == 'Random forests', 'pred'] = np.concatenate(
    [rf_quantile(rf_full, X_test, q) for q in QUANTILES]) 
# %%
preds.loc[preds.method == 'Gradient boosting', 'pred'] = \
    np.concatenate([gb_quantile(X_train, train_labels, X_test, q) 
                    for q in QUANTILES])
#%%
preds['quantile_loss'] = quantile_loss(preds.q, preds.label, 
                                            preds.pred)
plot_loss_comparison(preds)
# %%
preds.loc[preds.method == 'Random forests', 'pred']
# %%
tt = get_test(cons, unit, removed_cols)
tt = (tt - mean) / std
tt = sm.add_constant(tt)
s = np.concatenate(
    [rf_quantile(rf_full, tt, q).reshape(-1, 1) for q in QUANTILES], axis=1)

# %%
t1 = np.load('./target1.npy')
t2 = np.load('./target2.npy')

d1 = pd.DataFrame(data=t1, columns=QUANTILES)
d2 = pd.DataFrame(data=t2, columns=QUANTILES)
# %%
submission = pd.read_csv('./data/sample_submission.csv')
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = d1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = d2.sort_index().values
submission.to_csv(f'./submission/rf.csv', index=False)

# %%
