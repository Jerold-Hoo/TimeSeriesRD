
# -*- coding: utf-8 -*-

# @File       : LSTM.py
# @Date       : 2019-05-31
# @Author     : Jerold
# @Description: for the test of WYNY


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime
import WYNYtest

import warnings
warnings.filterwarnings('ignore')

MY_PATH = r"G:\111MyData\test_data201803.csv"
MY_MODEL_PATH = r"G:\models"
SAVE_MODEL_PATH = r"G:\models\LSTM.model"

# 定义参数
rnn_unit=20       #hidden layer units
input_size=4
output_size=1
lr=0.0006        #学习率
tf.reset_default_graph()
#输入层、输出层权重、偏置

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
}

def get_data(data,batch_size=60, time_step=5, train_begin=0, train_end=7104):
    batch_index = []

    #scaler_for_y = MinMaxScaler(feature_range=(0, 1))
    #scaler_for_data = MinMaxScaler(feature_range=(0, 1))  # 按列做minmax缩放
    mean = data[:,0].mean(axis=0)
    std = data[:,0].std(axis=0)
    scaler = StandardScaler().fit(data)
    data = scaler.transform(data)

    X_train = data[:train_end-1,:]
    Y_train = data[1:train_end,0]

    X_test = data[train_end-time_step:-1,:]
    Y_test = data[train_end:,0]

    train_x, train_y = [], []  # 训练集x和y初定义
    for i in range(len(X_train)-time_step):
        # 标记 batch
        if i % batch_size == 0:
            batch_index.append(i)
        x = X_train[i:i + time_step]
        train_x.append(x.tolist())
        #y = label_train[i:i + time_step, np.newaxis]
        y = Y_train[i:i + time_step,np.newaxis]
        train_y.append(y.tolist())

    batch_index.append((len(X_train) - time_step))

    # 测试数据
    test_x = []
    for i in range(len(Y_test)):
        x = X_test[i:i + time_step]
        y = X_test[i]
        test_x.append(x.tolist())
    test_y = Y_test

    return batch_index, train_x, train_y, test_x, test_y,[mean,std]

def lstm(X):
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入

    cell=tf.contrib.rnn.BasicLSTMCell(rnn_unit)
    #cell=tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

def train_lstm(data,batch_size=80, time_step=15, train_begin=0, train_end=7102):

    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y, test_x, test_y, scaler = get_data(data,batch_size, time_step, train_begin,
                                                                           train_end)

    # 定义lstm网络
    pred, _ = lstm(X)

    # 损失函数
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    # 保存模型
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 重复训练5000次
        iter_time = 5000
        for i in range(iter_time):
            for step in range(len(batch_index) - 1):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                 Y: train_y[batch_index[step]:batch_index[step + 1]]})
            if i % 10 == 0:
                print('iter:', i, 'loss:', loss_)

        saver.save(sess, SAVE_MODEL_PATH, global_step=i)

        ####predict####
        test_predict = []

        for step in range(len(test_x)):
            #print(step,':',np.shape([test_x[step]]))
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            predict = prob.reshape((-1))
            test_predict.append(predict[-1])

        mean = scaler[0]
        std = scaler[1]
        test_y = np.mat(test_y)*std + mean
        test_predict = np.mat(test_predict)*std + mean

        #test_y = scaler_for_y.inverse_transform(np.mat(test_y).T)
        #test_predict = scaler_for_y.inverse_transform([[i] for i in test_predict])

        #rmse = np.sqrt(mean_squared_error(test_predict, test_y))
        mae = mean_absolute_error(y_pred=test_predict, y_true=test_y)

        print('mae:', mae)
    return test_y,test_predict

# 使用模型
def prediction(data, batch_size=60, time_step=5, train_begin=0, train_end=7104):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    batch_index, train_x, train_y, test_x, test_y, scaler = get_data(data,batch_size, time_step, train_begin, train_end)
    pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint(MY_MODEL_PATH)
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x)):
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            predict = prob.reshape((-1))
            test_predict.append(predict[-1])

        mean = scaler[0]
        std = scaler[1]

        mean = scaler[0]
        std = scaler[1]
        test_y = np.mat(test_y)*std + mean
        test_predict = np.mat(test_predict)*std + mean
        mae = mean_absolute_error(y_pred=test_predict, y_true=test_y)
        mape = mae / test_y.mean() * 100
        print('mape:',mape,'mae:',mae)

    return test_y,test_predict

def program():
    data = pd.read_csv(MY_PATH,index_col=0)  # 读入数据
    data.loc[:, 'time'] = pd.to_datetime(data['time'])
    data['day'] = [i.day for i in data['time']]
    data['weekday'] = [i.dayofweek for i in data['time']]
    data['hour'] = [i.hour for i in data['time']]

    #剔除异常值
    # 通过 XGboost算法的判断0值模型来判断
    data = data[data['KWH'] > 2100]
    train_end = len(data[data['time'] < datetime(2006,12,1)]) + 1

    data = data.iloc[:,1:].astype(np.float64).values

    # 训练模型请使用以下语句
    #test_y, test_predict = train_lstm(data,batch_size=200, time_step=10, train_begin=0, train_end=train_end)


    test_y, test_predict = prediction(data, batch_size=200, time_step=10, train_begin=0, train_end=train_end)
    plt.figure(figsize=(24, 8))
    plt.plot(test_predict.tolist()[0])
    plt.plot(test_y.tolist()[0])
    plt.title("Use LSTM to predict 'KWH' data in 2016.12")
    plt.legend(['predict-KWH','KWH'])
    plt.show()

if __name__ == "__main__":
    program()
