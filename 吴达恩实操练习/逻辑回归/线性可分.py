import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#1获取数据
data = pd.read_csv('./ex2data1.txt',names=['exam1','exam2','Accecpted'])
# datafram前5个数据
print(data.head())

#2图像绘制
# fig,ax = plt.subplots()
# ax.scatter(data[data['Accecpted']==0]['exam1'],data[data['Accecpted']==0]['exam2'],c='r',marker='x',label='y=0')
# ax.scatter(data[data['Accecpted']==1]['exam1'],data[data['Accecpted']==1]['exam2'],c='b',marker='o',label='y=1')
# ax.set(
#     xlabel='exam1',
#     ylabel='exam1',
# )
# plt.show()

#3 数据集
def get_Xy(data):
    data.insert(0,'ones',1)
    X_ = data.iloc[:,0:-1]
    X = X_.values
    y_ = data.iloc[:,-1]
    y = y_.values.reshape(len(y_),1)
    return X, y

X, y = get_Xy(data)
print('x shape',X.shape,'y shape', y.shape)

#4 cost函数
#定义sigmoid函数
def sigmoid(z): 
    return 1 / (1 + np.exp(-z))

def costFunction(X, y, theta):
    predictY = sigmoid(X @ theta)
    # print('predictY',predictY)
    # print('y',y)
    #注意是点乘，不是矩阵乘法
    first = y * np.log(predictY)
    second = (1-y) * np.log(1-predictY)

    #注意累加 需要用np.sum
    return -np.sum(first + second) / len(y)

theta = np.zeros((3,1))
cost_init = costFunction(X, y, theta)
print('cost_init', cost_init)

#5梯度下降
def gradientDescent(X,y,theta,alpha,iters):
    costs = []
    for i in range(iters):
        predictY = sigmoid(X @ theta)
        theta = theta - (X.T @ (predictY - y)) * alpha / len(X)
        cost = costFunction(X, y, theta)
        costs.append(cost)
        
        if i%1000 == 0:
            print(i,cost)
    return theta, costs

alpha = 0.004
iters = 200000
theta_final, costs = gradientDescent(X,y,theta,alpha,iters)

#6 决策边界
coef1 = -theta_final[0,0]/theta_final[2,0]
coef2 = -theta_final[1,0]/theta_final[2,0]
x = np.linspace(20,100,100)
f = coef1 + coef2*x
fig,ax = plt.subplots()
ax.scatter(data[data['Accecpted']==0]['exam1'],data[data['Accecpted']==0]['exam2'],c='r',marker='x',label='y=0')
ax.scatter(data[data['Accecpted']==1]['exam1'],data[data['Accecpted']==1]['exam2'],c='b',marker='o',label='y=1')
ax.set(
    xlabel='exam1',
    ylabel='exam1',
)
ax.plot(x, f, c='g')
plt.show()
