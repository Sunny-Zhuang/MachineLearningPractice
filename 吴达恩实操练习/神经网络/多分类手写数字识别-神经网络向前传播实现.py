import numpy as np
import scipy.io as sio


#获取数据
data = sio.loadmat('./ex3data1.mat')
# print(data)
raw_X = data['X']
raw_y = data['y']

X = np.insert(raw_X,0,values=1,axis=1)
y = raw_y.flatten()

#获取theta权重参数
data = sio.loadmat('./ex3weights.mat')
theta1 = data['Theta1']
theta2 = data['Theta2']
print(theta1.shape)
print(theta2.shape)

# 开始向前传播
def sigmoid(z): 
    return 1 / (1 + np.exp(-z))
a1 = X
z2 = X @ theta1.T
a2 = sigmoid(z2)
#加入偏执项
a2 = np.insert(a2,0,values=1,axis=1)
print('a2', a2.shape)

z3 = a2 @ theta2.T
a3 = sigmoid(z3)
print('a3', a3.shape)

#预测
pred = np.argmax(a3,axis=1)
pred = pred + 1
acc = np.mean(pred == y)
print(acc)