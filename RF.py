#%%
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import ensemble
import statsmodels.api as sm

from matplotlib.ticker import FuncFormatter

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

# sns.set_palette(sns.color_palette('Blues', len(QUANTILES)))
sns.set_palette(sns.color_palette('Blues'))
# Set dots to a light gray
dot_color = sns.color_palette('coolwarm', 3)[1]
# %%
from module.data import get_train, get_test, save_submission

cons = 4
unit = 48
removed_cols = ['Day', 'Hour', 'Minute']

df_train_x, df_train_y = get_train(cons, unit, removed_cols)
test_df = get_test(cons, unit, removed_cols)

X_train_1 = df_train_x
X_train_2 = df_train_x.copy(deep=True)
Y_train_1 = df_train_y.iloc[:, 0]
Y_train_2 = df_train_y.iloc[:, 1]

# without kfold
from sklearn.model_selection import train_test_split
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(df_train_x, df_train_y.iloc[:, 0], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(df_train_x, df_train_y.iloc[:, 1], test_size=0.3, random_state=0)
xval = (X_valid_1.copy(deep=True), X_valid_2.copy(deep=True) )
yval = (Y_valid_1.copy(deep=True), Y_valid_2.copy(deep=True) )

# max idx = cons - 1
def denorm(df, idx):
    x = df.iloc[:, idx]
    return x * std[idx] + mean[idx]

def preprocess(train_df, test_df, valid_df=None):
    # Normalize
    mean = train_df.mean(axis=0)
    std = train_df.std(axis=0)
    train_df = (train_df - mean) / std
    # valid_df = (valid_df - mean) / std
    test_df = (test_df - mean) / std

    X_train = sm.add_constant(train_df)
    # X_valid = sm.add_constant(valid_df)
    X_test = sm.add_constant(test_df)
    return X_train, X_test

X_train_1, X_test_1 = preprocess(X_train_1, test_df.copy(deep=True))
X_train_2, X_test_2 = preprocess(X_train_2, test_df.copy(deep=True))

# %%
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# QUANTILES.reverse()  # Test out to see if we're getting different results.

quantiles_legend = [str(int(q * 100)) + 'th percentile' for q in QUANTILES]
# %%
METHODS=['Random forests 1', 'Random forests 2']
preds = np.array([(method, q, x) 
                  for method in METHODS 
                  for q in QUANTILES
                  for x in X_valid_1.const])
preds = pd.DataFrame(preds)
preds.columns = ['method', 'q', 'x']
preds = preds.apply(lambda x: pd.to_numeric(x, errors='ignore'))

preds['label'] = pd.concat([pd.concat([yval[0]]*9), pd.concat([yval[1]]*9)]).reset_index(drop=True)
preds = preds[['method', 'q', 'label']]
# %%
# pandas version rather than Keras.
def quantile_loss(q, y, f):
    # q: Quantile to be evaluated, e.g., 0.5 for median.
    # y: True value.
    # f: Fitted or predicted value.
    e = y - f
    return np.maximum(q * e, (q - 1) * e)

def rf_quantile(m, X, q):
    rf_preds = []
    for estimator in m.estimators_:
        rf_preds.append(estimator.predict(X))
    rf_preds = np.array(rf_preds).transpose()  # One row per record.
    return np.percentile(rf_preds, q * 100, axis=1)

def get_rf(X_train, train_labels):
    N_ESTIMATORS = 2000
    rf = ensemble.RandomForestRegressor(n_estimators=N_ESTIMATORS, 
                                        min_samples_leaf=1, random_state=3, 
                                        verbose=True, 
                                        n_jobs=-1)  # Use maximum number of cores.
    rf.fit(X_train, train_labels)
    return rf
# %%
# using kfold
# from sklearn.model_selection import KFold

# def kfold_train(Xtrain, Ytrain):
#     model = ensemble.RandomForestRegressor(n_estimators=500, 
#                                             min_samples_leaf=1, random_state=3, 
#                                             verbose=True, 
#                                             n_jobs=-1)

#     kfold = KFold(n_splits=4, shuffle=True, random_state=0)

#     fold = 0
#     for train_index, test_index in kfold.split(Xtrain,Ytrain):
#         X_train, X_test = Xtrain.iloc[train_index,:], Xtrain.iloc[test_index, :]
#         Y_train, Y_test = Ytrain.iloc[train_index], Ytrain.iloc[test_index]
        
#         model.fit(X_train, Y_train)
#         losses = []
#         for q in QUANTILES:
#             pred = rf_quantile(model, X_test, q)
#             loss = quantile_loss(q, Y_test, pred)
#             losses.append(loss.mean())
#         print(fold, ':')
#         print(np.mean(losses), losses)
#     return model

# model1 = kfold_train(X_train_1, Y_train_1)
# model2 = kfold_train(X_train_2, Y_train_2)

# s1 = np.concatenate(
#         [rf_quantile(model1, X_test_1, q).reshape(-1, 1) for q in QUANTILES], axis=1)
# s2 = np.concatenate(
#         [rf_quantile(model2, X_test_2, q).reshape(-1, 1) for q in QUANTILES], axis=1)

# results_1 = pd.DataFrame(data=s1, columns=QUANTILES)
# results_2 = pd.DataFrame(data=s2, columns=QUANTILES)
# %%
# without kfold
rf1 = get_rf(X_train_1, Y_train_1)
rf2 = get_rf(X_train_2, Y_train_2)

preds.loc[preds.method == 'Random forests 1', 'pred'] = np.concatenate(
    [rf_quantile(rf1, X_valid_1, q) for q in QUANTILES])

print(0)
preds.loc[preds.method == 'Random forests 2', 'pred'] = np.concatenate(
    [rf_quantile(rf2, X_valid_2, q) for q in QUANTILES])

# results_1, results_2 = ??
# %%
# DRAW QUANTILE LOSS
for i, method in enumerate(METHODS):
    num = 1
    ax = plt.scatter(xval[i].iloc[:, num], yval[i], color=dot_color)
    preds['x_denorm'] = pd.concat([xval[i].iloc[:, num]]*len(QUANTILES)*len(METHODS)).reset_index(drop=True)
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
        lambda y, _: '{}'.format(y)))
    plt.xlabel('Proportion of owner-occupied units built prior to 1940')
    plt.ylabel('Median value of owner-occupied homes')
    plt.title(method + f'{xval[i].columns[num]} vs. Target', loc='left')
    sns.despine(left=True, bottom=True)
    plt.show()
# %%
# COMPARE TARGET1 MODEL AND TARGET2 MODEL
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

preds['quantile_loss'] = quantile_loss(preds.q, preds.label, 
                                            preds.pred)
plot_loss_comparison(preds)

# %%
submission = pd.read_csv('./data/sample_submission.csv')
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = d1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = d2.sort_index().values
submission.to_csv(f'./submission/rf_@@.csv', index=False)


# %%
