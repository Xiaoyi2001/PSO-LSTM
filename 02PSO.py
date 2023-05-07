# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM,Dropout,Dense
from scipy.io import loadmat,savemat

np.random.seed(0)
tf.random.set_seed(10)
time_steps=5
# 定义model
def build_model(hid_dim1,hid_dim2,fc):
    model = tf.keras.Sequential([
        LSTM(hid_dim1, input_shape=(time_steps,1),return_sequences=True),  # 第一层lstm神经元数
        LSTM(hid_dim2, return_sequences=False),  # 第二层lstm神经元数
        Dense(fc,activation="relu"),  # 全连接层数
        Dense(1)
    ])
    return model

def fitness(pop,P,T,Pt,Tt):
    # 共优化6个参数
    epochs_num=int(pop[0])    #迭代次数
    batch_size=int(pop[1])    #Batchsize
    lstm_out_dim1=int(pop[2]) #第一层lstm的输出神经元数
    lstm_out_dim2=int(pop[3]) #第一层lstm的输出神经元数
    fc=int(pop[4])            #全连接层的神经元数
    lr=pop[5]                 #学习率
    model=build_model(lstm_out_dim1,lstm_out_dim2, fc)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr), metrics=['mean_squared_error'])
    model.fit(P, T,epochs=epochs_num, batch_size=batch_size, verbose=0,
                      validation_data=(Pt,Tt))
    yhat = model.predict(Pt, verbose=0) #预测
    return np.mean(np.square((yhat-Tt)))

def PSO(x_train,y_train,x_test,y_test):
    #PSO参数设置
    max_iter = 10  # 最大迭代次数(根据情况修改)

    dim=6
    #搜索参数个数：
    # 迭代次数 batchsize   lstm1 lstm2  全连接层的神经元 学习率
    lb=[10,   10,   10,   10,  10,  0.001]
    ub=[60,  100,  100, 100,  100, 0.01]# 6个参数的上下界范围
    pN=5                               #粒子数
    c1 = 1.5;c2 = 1.6;w=0.7
    #初始化
    X = np.zeros((pN,dim))
    V = np.zeros((pN,dim))
    pbest = np.zeros((pN,dim))
    gbest = np.zeros((1,dim))
    p_fit = np.zeros(pN)
    result=np.zeros((max_iter,dim))
    fit = np.inf
    for i in range(pN):
        for j in range(dim):
            if j==dim-1:
                X[i][j] = (ub[j]-lb[j])*np.random.rand()+lb[j]# 产生小数
            else:
                X[i][j] = np.random.randint(lb[j],ub[j])
            V[i][j] = np.random.rand()
        pbest[i] = X[i].copy()
        tmp = fitness(X[i,:],x_train,y_train,x_test,y_test)
        p_fit[i] = tmp.copy()
        if(tmp < fit):
            fit = tmp.copy()
            gbest = X[i].copy()
    # 开始循环迭代
    trace=[]
    for t in range(max_iter):
        for i in range(pN):
            r1=np.random.rand()
            r2=np.random.rand()
            V[i,:] = w*V[i,:] + c1*r1*(pbest[i] - X[i,:])+c2*r2*(gbest - X[i,:])
            X[i,:] = X[i,:] + V[i,:]

        X=boundary(X,lb,ub)#边界判断
        for i in range(pN): #更新gbest\pbest
            temp = fitness(X[i,:],x_train,y_train,x_test,y_test)
            if(temp < p_fit[i]):      #更新个体最优
                p_fit[i] = temp.copy()
                pbest[i,:] = X[i,:].copy()
                if(p_fit[i] < fit):  #更新全局最优
                    gbest = X[i,:].copy()
                    fit = p_fit[i].copy()
        print ("iteration",t+1,"=",fit,[gbest[i] if i==dim-1 else int(gbest[i]) for i in range(len(gbest))])
        result[t,:]=gbest.copy()
        trace.append(fit)
    return trace ,gbest,result

def boundary(pop,lb,ub):
    for i in range(pop.shape[0]):
        # 除学习率外 其他的都为整数
        dim=pop.shape[1]
        for j in range(dim):
            if j==dim-1:
                pop[i,j]=pop[i,j]
            else:
                pop[i,j]=int(pop[i,j])
        # 防止粒子跳出范围
        for j in range(dim):
            if j==dim-1:
                if pop[i,j]>ub[j] or pop[i,j]<lb[j]:
                    pop[i,j] = (ub[j]-lb[j])*np.random.rand()+lb[j]
            else:
                if pop[i,j]>ub[j] or pop[i,j]<lb[j]:
                    pop[i,j] = np.random.randint(lb[j],ub[j])
    return pop

def testing(model,x_test):
    #模型进行预测
    pred_data = model.predict(x_test)
    return pred_data

def data_load():
    data = pd.read_csv('data/data.csv').values
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

#模型评估mae
def eav(y_test,pred_test):
    test_mae = np.mean(np.abs(y_test - pred_test))
    print('mae:',test_mae)

def fig(trace,result):
    # 绘图属性
    plt.rcParams['font.sans-serif'] = ['SimHei']

    #PSO适应度曲线图
    plt.figure()
    plt.plot(trace)
    plt.xlabel('iteration')
    plt.ylabel('适应度值')
    plt.title("PSO适应度曲线图")
    plt.savefig('./figure/PSO适应度曲线图.png', dpi=500)
    plt.show()

    #绘制子图
    plt.figure()
    plt.subplot(3, 2, 1)
    plt.plot(result[:, 0])
    plt.xlabel('进化代数')
    plt.ylabel('迭代次数')

    plt.subplot(3, 2, 2)
    plt.plot(result[:, 1])
    plt.xlabel('进化代数')
    plt.ylabel('batchsize')

    plt.subplot(3, 2, 3)
    plt.plot(result[:, 2])
    plt.xlabel('进化代数')
    plt.ylabel('第一层LSTM神经元')

    plt.subplot(3, 2, 4)
    plt.plot(result[:, 3])
    plt.xlabel('进化代数')
    plt.ylabel('第二层LSTM神经元')

    plt.subplot(3, 2, 5)
    plt.plot(result[:, 4])
    plt.xlabel('进化代数')
    plt.ylabel('FC层神经元数')

    plt.subplot(3, 2, 6)
    plt.plot(result[:, 5])
    plt.xlabel('进化代数')
    plt.ylabel('学习率')
    plt.savefig('./figure/PSO各参数优化曲线.png', dpi=500)
    plt.show()

if __name__=='__main__':
    #加载数据，划分数据集，归一化
    x_train,y_train,x_test,y_test,ss_Y=data_load()
    print("             [迭代次数    Batchsize   第一层lstm数 第二层lstm数  全连接层   学习率]")
    #调用函数返回优化参数与适应度
    trace, best, result = PSO(x_train,y_train,x_test,y_test)
    result = np.array(result)  #转换为numpy类型
    #把参数与适应度 保存到.mat文件
    savemat('pso_para.mat',{'para_trace':trace,'para_best':best,'para_result':result})
    para=loadmat('pso_para.mat')   #加载矩阵文件

    trace=para['para_trace'].reshape(-1,) #获取trace并转换成一维列表
    result = para['para_result']   #获取参数
    fig(trace, result)   #绘图


