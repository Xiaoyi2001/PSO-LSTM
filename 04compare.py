# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#说明
# 01LSTM是未优化的
# 02PSO.py 优化保存参数
# 03PSO-LSTM.py 加载优化的参数预测
# 04compare 对比优化前后


#模型评估
def eav(name,y_test,pred_test):
    test_mae = np.mean(np.abs(y_test - pred_test))
    print(name,'mae:',test_mae)

data0=np.load('result/lstm_result.npz')['true'].reshape(-1,1)
data1=np.load('result/lstm_result.npz')['pred'].reshape(-1,1)
data2=np.load('result/pso_result.npz')['pred'].reshape(-1,1)

#评估
eav("lstm",data1,data0)
eav("psolstm",data2,data0)

#绘图
plt.rcParams['font.sans-serif'] = ['SimHei'] #黑体
plt.figure()
plt.plot(data0, c='k', label='实际值')
plt.plot(data1, c='b', label='LSTM')
plt.plot(data2, c='r', label='PSO-LSTM')
plt.legend()
plt.xlabel('样本点')
plt.ylabel('功率')
plt.title('结果对比')
plt.savefig('./figure/结果对比.jpg')
plt.show()













