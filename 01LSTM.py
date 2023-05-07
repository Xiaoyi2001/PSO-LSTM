# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM,Dropout,Dense
from scipy.io import loadmat,savemat

# 设置全局 Python 随机种子
np.random.seed(1)
# 设置 TensorFlow 随机种子
tf.random.set_seed(1)

time_steps = 5
#定义模型
def build_model(x_train,y_train,x_test,y_test):
    model = tf.keras.Sequential([
        LSTM(80,input_shape=(time_steps,1),return_sequences=True),   #第一层lstm神经元数
        LSTM(40, return_sequences=False),                            #第二层lstm神经元数
        Dense(6,activation="relu"),                                  # 全连接层数
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),#学习率
                  loss="mean_squared_error")#回归问题，损失函数使用均方误差
    #训练模型
    history = model.fit(x_train,
              y_train,
              batch_size=32,    #batchsize
              epochs=20,        #迭代的次数
              validation_data=(x_test,y_test), #验证集
              validation_freq=1)
    #模型显示计算过程
    model.summary()
    return model

def testing(model,x_test):
    #测试集输入模型进行预测
    pred_data = model.predict(x_test)
    return pred_data

def data_load():

    data = pd.read_csv('data/mndata.csv').values
    row = data.shape[0]
    num_train = int(row * 0.8)  # 训练集

    # 训练集与测试集划分
    x_train = data[:num_train, 0:10]
    y_train = data[:num_train, 10:11]

    x_test = data[num_train:, 0:10]
    y_test = data[num_train:, 10:11]

    # 归一化
    ss_X = StandardScaler().fit(x_train)
    ss_Y = StandardScaler().fit(y_train)
    #转换为三维
    x_train = ss_X.transform(x_train).reshape(x_train.shape[0], time_steps, -1)
    y_train = ss_Y.transform(y_train)
    # 转换为三维
    x_test = ss_X.transform(x_test).reshape(x_test.shape[0], time_steps, -1)
    y_test = ss_Y.transform(y_test)
    return x_train,y_train,x_test,y_test,ss_Y

#模型评估并保存数据
def eav(y_test,pred_test):
    test_mae = np.mean(np.abs(y_test - pred_test))
    #保存数据供对比
    np.savez('result/lstm_result.npz', true=y_test, pred=pred_test)
    print('mae:',test_mae)

if __name__=='__main__':
    #加载数据并归一化
    x_train,y_train,x_test,y_test,ss_Y=data_load()
    #构建模型
    model = build_model(x_train,y_train,x_test,y_test)
    #预测
    pred_data = testing(model, x_test)

    #反归一化
    y_test =   ss_Y.inverse_transform(y_test)
    pred_test = ss_Y.inverse_transform(pred_data)

    # 模型评估并保存数据
    eav(y_test, pred_test)

    # 绘图
    #绘图属性
    plt.rcParams['font.sans-serif'] = ['SimHei'] #设置字体防止中文乱码
    plt.figure()
    plt.plot(y_test, c='k',label='实际值')
    plt.plot(pred_test, c='r',label='预测值')
    plt.legend()
    plt.xlabel('样本点')
    plt.ylabel('功率')
    plt.title('LSTM预测对比')
    plt.savefig('./figure/LSTM预测对比.jpg')
    plt.show()


