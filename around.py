# %%
from sklearn.model_selection import train_test_split
from module import *

def concat_data(data): # num of consecutive days
    temp = data.copy()
    res = data.copy()
    cols_name = res.columns
    res.columns = [f'{col}_{0}' for col in cols_name]
    for ele in [48, 96, 144, 192]:
        new_ele = temp.shift(-ele, fill_value = np.nan)
        new_ele.columns = [f'{col}_{ele}' for col in cols_name]
        res = pd.concat([res, new_ele],axis=1)
    return res

def preprocess_data(data, consecutive, unit, is_train=True):
    # 원하는 칼럼 추가는 여기서
    # ex, temp['GHI'] = temp['DHI'] + temp['DNI']
    temp = data.copy()
    removed_cols = ['Day', 'Hour', 'Minute']

    temp = temp.drop(removed_cols, axis='columns')
    # temp = concat_data(temp, consecutive, unit)
    temp = concat_data(temp)
    temp = temp.fillna(method='bfill')

    if is_train:
        after_days = [1, 2]
        col = f'TARGET_192'

        temp = add_future_feats(temp, after_days, col)
        temp = temp.dropna()
        return temp

    else:
        temp = get_one_data(temp)
        return temp

cons = 2
unit = 1

temp = pd.read_csv('./data/train/train.csv')
df_train = preprocess_data(temp, cons, unit, is_train=True)
df_train.to_pickle('./df_train.pkl')


X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], test_size=0.3, random_state=0)

df_test = []

for i in range(81):
        file_path = './data/test/' + str(i) + '.csv'
        temp = pd.read_csv(file_path)
        temp = preprocess_data(temp, cons, unit, is_train=False)
        df_test.append(temp)

X_test = pd.concat(df_test)
# %%
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
from lightgbm import LGBMRegressor

# Get the model and the predictions in (a) - (b)
def LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test):

    # (a) Modeling  
    model = LGBMRegressor(objective='quantile', alpha=q,
                         n_estimators=10000, bagging_fraction=0.7, learning_rate=0.027, subsample=0.7)                   
                         
    model.fit(X_train, Y_train, eval_metric = ['quantile'], 
          eval_set=[(X_valid, Y_valid)], early_stopping_rounds=100, verbose=500)

    # (b) Predictions
    pred = pd.Series(model.predict(X_test).round(2))
    return pred, model
# %%
# Target 예측

def train_data(X_train, Y_train, X_valid, Y_valid, X_test):

    LGBM_models=[]
    LGBM_actual_pred = pd.DataFrame()

    for q in quantiles:
        print(q)    
        pred , model = LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test)
        LGBM_models.append(model)
        LGBM_actual_pred = pd.concat([LGBM_actual_pred,pred],axis=1)

    LGBM_actual_pred.columns=quantiles
    
    return LGBM_models, LGBM_actual_pred

# %%
# Target1
models_1, results_1 = train_data(X_train_1, Y_train_1, X_valid_1, Y_valid_1, X_test)
results_1.sort_index()[:48]

# Target2
models_2, results_2 = train_data(X_train_2, Y_train_2, X_valid_2, Y_valid_2, X_test)
results_2.sort_index()[:48]

# %%
from  lightgbm import plot_importance
for model in models_1:
    plot_importance(model)
# %%
submission = pd.read_csv('./data/sample_submission.csv')
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values
submission.to_csv('./submission/submission.csv', index=False)
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
