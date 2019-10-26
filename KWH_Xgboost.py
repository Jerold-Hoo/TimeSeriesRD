# -*- coding: utf-8 -*-

# @File       : XGboost.py
# @Date       : 2019-05-29
# @Author     : Jerold
# @Description: for the test of WYNY

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,roc_curve,auc,precision_score,recall_score,accuracy_score
from datetime import datetime

MY_PATH = r"G:\111MyData\test_data201803.csv"

def check_data(data):
    train = data[data['time'] < datetime(2006,12,1)]
    to_predict = data[data['time'] >= datetime(2006,12,1)]

    data[['KWH']].boxplot()
    plt.show()

def SearchCV(X_train, X_test, Y_train, Y_test):

    # 1 确定 n_estimators
    cv_params = {'n_estimators': [2000, 3000]}
    other_params = {'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    mod = xgb.XGBRegressor(**other_params)
    g_serch = GridSearchCV(estimator=mod,param_grid=cv_params)
    g_serch.fit(X_train,Y_train)

    #mod = xgb.XGBRegressor(**{**other_params,**g_serch.best_params_})
    #mod.fit(X_train,Y_train)
    print(g_serch.best_score_)
    print(g_serch.best_params_)
    print(g_serch.best_estimator_.score(X_train,Y_train))
    print(g_serch.best_estimator_.score(X_test, Y_test))
    print(g_serch.best_estimator_.feature_importances_)
    return

def use_xgboost_judge_zeros(X_train, X_test, Y_train, Y_test):
    params = {'booster': 'gbtree',
              'n_estimators': 500,
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.75,
              'colsample_bytree': 0.75,
              'min_child_weight': 1,
              'eta': 0.025,
              'seed': 0 }

    mod = xgb.XGBClassifier(**params)
    mod.fit(X_train,Y_train)

    Y_pre = mod.predict(X_test)
    y_score = mod.predict_proba(X_test)
    fpr,tpr,thresholds = roc_curve(Y_test, y_score[:,1])
    auc_s = auc(fpr,tpr)
    accuracy_s = accuracy_score(Y_test,Y_pre)
    precision_s = precision_score(Y_test, Y_pre)
    recall_s = recall_score(Y_test, Y_pre)
    print('auc:',auc_s,'accuracy:',accuracy_s,'precision:',precision_s,'recall:',recall_s)
    print(pd.Series(mod.feature_importances_, index=X_train.columns).sort_values(ascending=False))
    return mod

def use_xgboost(X_train, X_test, Y_train, Y_test):

    params = {'n_estimators':500,
              'learning_rate': 0.1,
              'max_depth': 5,
              'min_child_weight': 1,
              'lambda': 10,
              'seed': 0,
              'subsample': 0.6,
              'colsample_bytree': 0.8,
              'gamma': 0.1,
              'reg_alpha': 0,
              'reg_lambda': 1}

    mod = xgb.XGBRegressor(**params)
    mod.fit(X_train,Y_train)
    print(mod.score(X_train,Y_train))
    print(mod.score(X_test, Y_test))
    print(pd.Series(mod.feature_importances_,index=X_train.columns).sort_values(ascending=False))
    return mod

# 预处理数据
def deal_data_forXgBoost(data,fronts=3):

    data.loc[:,'time'] = pd.to_datetime(data['time'])
    #data['month'] = [i.month for i in data['time']]
    #data['day'] = [i.day for i in data['time']]
    data['weekday'] = [i.dayofweek for i in data['time']]
    data['hour'] = [i.hour for i in data['time']]

    # 增加前序时间作为特征 fronts 参数确定增加多少个前序时间
    n_data = len(data)
    for i in range(1,fronts+1,1):
        front = data['KWH'][0:n_data-i]
        front.index = front.index + i
        new_c = pd.Series([0] * i).append(front)
        data['front'+str(i)] = new_c

    # 确定 X的名称和Y的名称
    x_names = data.columns.tolist().copy()
    x_names.remove('time')
    x_names.remove('KWH')
    y_name = 'KWH'

    # 处理用于 二分类是否为异常的数据集
    data_classf_z = data.copy()
    data_classf_z.loc[:, 'KWH'] = data_classf_z.loc[:,'KWH'].where(data_classf_z['KWH']>2100,1)
    data_classf_z.loc[:, 'KWH'] = data_classf_z.loc[:, 'KWH'].where(data_classf_z['KWH'] == 1,0)
    train_b = data_classf_z[data_classf_z['time'] < datetime(2006,12,1)]
    to_predict_b = data_classf_z[data_classf_z['time'] >= datetime(2006,12,1)]

    X_train_b, X_test_b, Y_train_b, Y_test_b = train_test_split(train_b[x_names], train_b[y_name], test_size=0.25, random_state=33)

    # 处理用于回归的数据
    # 小于 2000的异常数据，使用前一个非异常数据填充
    data.iloc[:, 4:] = data.iloc[:,4:].where(data.iloc[:,4:]>2100,pd.NaT)
    data = data.fillna(method='pad')
    data = data[16:]
    data.iloc[:, 4:] = data.iloc[:,4:].astype('float64')

    train = data[data['time'] < datetime(2006,12,1)]
    to_predict = data[data['time'] >= datetime(2006,12,1)]

    #剔除异常值
    train = train[train['KWH'] > 2100]
    X_train, X_test, Y_train, Y_test = train_test_split(train[x_names], train[y_name], test_size=0.25, random_state=33)

    return X_train, X_test, Y_train, Y_test, to_predict[x_names], to_predict[y_name],X_train_b, X_test_b, Y_train_b, Y_test_b, to_predict_b[x_names], to_predict_b[y_name]

# 评价二分类模型
def evaluate_mod_binary(mod,X_to_predict_b,Y_to_predict_b):

    Y_pre = mod.predict(X_to_predict_b)
    y_score = mod.predict_proba(X_to_predict_b)
    fpr,tpr,thresholds = roc_curve(Y_to_predict_b, y_score[:,1])
    auc_s = auc(fpr,tpr)
    accuracy_s = accuracy_score(Y_to_predict_b,Y_pre)
    precision_s = precision_score(Y_to_predict_b, Y_pre)
    recall_s = recall_score(Y_to_predict_b, Y_pre)
    print('auc:',auc_s,'accuracy:',accuracy_s,'precision:',precision_s,'recall:',recall_s)
    #show = pd.DataFrame(Y_to_predict_b)
    #show['predict'] = y_score[:,1]
    #print(show[show['KWH']==1])
    return Y_pre

# 评价回归模型
def evaluate_mod_reg(mod,X_to_predict,Y_to_predict):

    predict_Y = mod.predict(X_to_predict)
    print(mean_absolute_error(Y_to_predict.values,predict_Y))
    return predict_Y

# 主运行函数
def program():

    data = pd.read_csv(MY_PATH,index_col=0)

    X_train, X_test, Y_train, Y_test, X_to_predict, Y_to_predict,X_train_b, X_test_b, Y_train_b, Y_test_b, X_to_predict_b,Y_to_predict_b = deal_data_forXgBoost(data,10)

    mod_c = use_xgboost_judge_zeros(X_train_b, X_test_b, Y_train_b, Y_test_b)
    predict_y_b = evaluate_mod_binary(mod_c,X_to_predict_b,Y_to_predict_b)

    print("---------------------------------------------------------")
    mod = use_xgboost(X_train, X_test, Y_train, Y_test)
    predict_y = evaluate_mod_reg(mod,X_to_predict,Y_to_predict)
    predict_y[predict_y_b == 1] = 0
    mae = mean_absolute_error(Y_to_predict, predict_y)
    print('MAPE:',mae / Y_to_predict.mean() * 100,'MAE:',mae)
    plt.figure()
    plt.plot(Y_to_predict.index, Y_to_predict)
    plt.plot(Y_to_predict.index, predict_y)
    plt.legend(['KWH','predict_KWH'])
    plt.title("Use XGboost to predict 'KWH' data in 2006.12")
    plt.show()

    #X_to_predict = X_to_predict[Y_to_predict > 2000]
    #Y_to_predict = Y_to_predict[Y_to_predict > 2000]


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    pd.set_option("display.max_columns",None)
    #pd.set_option("display.max_rows", None)

    program()

