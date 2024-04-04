import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 1.获取数据
data = pd.read_csv('./ex1data2.txt',names=['size','bedroom','price'])

#2 数据预处理： 归一化
def normalize_feature(data):
    return ( data - data.mean() ) / data.std()

data = normalize_feature(data)
# datafram前5个数据
print(data.head())

# 插入第一列全是1
data.insert(0,'ones',1)
# print(data.head())
# 切一下数据 :表示所有,第一个参数行，第二个参数列
X = data.iloc[:,0:-1]
print(X.head())
y = data.iloc[:,-1]
print(y.head())
# dataframe=》数组ndarray而是一种数据结构，用于存储和处理多维数据。
X = X.values
print(X.shape)
y = y.values
y=y.reshape(47,1)

# 4.损失函数
# 定义损失函数方法
def costFunction(X,y,theta):
    inner = np.power(X @ theta -y,2)
    return np.sum(inner)/(2 * len(X))

# 初始化theta
theta = np.zeros((3,1))
cost_init = costFunction(X, y, theta)
print('cost_init',cost_init)

#下降函数
def gradientDescent(X,y,theta,alpha,iters):
    costs = []
    for i in range(iters):
        theta = theta - (X.T @ (X @ theta - y)) * alpha / len(X)
        cost = costFunction(X, y, theta)
        costs.append(cost)
        
        if i%100 == 0:
            print(i,cost)
    return theta, costs

# alpha = 0.02
# candinate_alpha = [0.0003, 0.003, 0.03, 0.0001,0.001,0.01]
# iters = 2000
# # theta, costs = gradientDescent(X,y,theta,alpha,iters)

# # 画图看是否收敛
# fig,ax = plt.subplots()
# for alpha in candinate_alpha:
#     _,costs = gradientDescent(X,y,theta,alpha,iters)
#     ax.plot(np.arange(iters),costs,label=alpha)
#     ax.legend()

# ax.set(
#     xlabel='iters',
#     ylabel='costs',
#     title='iters vs costs'
# )
# plt.show()

# 拟合函数可视化

x = np.linspace(y.min(),y.max(),100)
y_ = theta[0,0] + theta[1,0] * x

fig,ax = plt.subplots()
ax.scatter(X[:,1],y,label="training data")
ax.plot(x,y_,'r',label="predict")
# scattor和plot同时显示
ax.legend()
ax.set(
    xlabel='population',
    ylabel='profit',
)
plt.show()

