import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 1.获取数据
data = pd.read_csv('./ex1data1.txt',names=['population','profit'])
# datafram前5个数据
# print(data.head())
# print(data.info())
# count mean等数据
# print(data.describe())

# 2.matplot画图
# 散点图
# data.plot.scatter('population','profit',label='population')
# plt.show()

# 3.准备数据集
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
print(y.shape)
# 把y编程二维数组
y = y.reshape(97,1)
print(y.shape)

# 4.正规方程
def normalEquation(X,y):
    theta =  np.linalg.inv(X.T@X)@X.T@y
    return theta

theta = normalEquation(X,y)
print('theta',theta)


# 拟合函数可视化

# x = np.linspace(y.min(),y.max(),100)
# y_ = theta[0,0] + theta[1,0] * x

# fig,ax = plt.subplots()
# ax.scatter(X[:,1],y,label="training data")
# ax.plot(x,y_,'r',label="predict")
# # scattor和plot同时显示
# ax.legend()
# ax.set(
#     xlabel='population',
#     ylabel='profit',
# )
# plt.show()









