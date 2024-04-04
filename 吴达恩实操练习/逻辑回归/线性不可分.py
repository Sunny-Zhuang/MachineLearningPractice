import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#1获取数据
data = pd.read_csv('./ex2data2.txt',names=['test1','test2','Accecpted'])
# datafram前5个数据
# print(data.head())

#2图像绘制
# fig,ax = plt.subplots()
# ax.scatter(data[data['Accecpted']==0]['test1'],data[data['Accecpted']==0]['test2'],c='r',marker='x',label='y=0')
# ax.scatter(data[data['Accecpted']==1]['test1'],data[data['Accecpted']==1]['test2'],c='b',marker='o',label='y=1')
# ax.set(
#     xlabel='exam1',
#     ylabel='exam1',
# )
# plt.show()

#3 线性不可分需要做特征映射
def  feature_mapping(x1,x2,power):
    data = {}

    for i in np.arange(power+1):
        for j in np.arange(i+1):
            # print('value',np.power(x1,i-j)*np.power(x2,j))
            # print('key','F{}{}'.format(i-j,j))
            data['F{}{}'.format(i-j,j)] = np.power(x1,i-j)*np.power(x2,j)
    
    return pd.DataFrame(data)

x1 = data['test1']
x2 = data['test2']
data2 = feature_mapping(x1,x2,6)
print(data2.head())


#4 构造数据集
X = data2.values
print(X.shape)
y = data.iloc[:,-1]
y = y.values
# 把y编程二维数组
y = y.reshape(len(y),1)
print(y.shape)

#5 损失函数
def sigmoid(z): 
    return 1 / (1 + np.exp(-z))

def costFunction(X, y, theta, lamda):
    predictY = sigmoid(X @ theta)
    #注意是点乘，不是矩阵乘法
    first = y * np.log(predictY)
    second = (1-y) * np.log(1-predictY)

    # l2正则化，由于特征映射可能会导致过拟合所以需要正则化
    reg = np.sum(np.power(theta[1:],2))*(lamda/(2*len(X)))

    #注意累加 需要用np.sum
    return -np.sum(first + second) / len(y) + reg

theta = np.zeros((28,1))
lamda = 1
cost_init = costFunction(X, y, theta, lamda)
print('cost_init', cost_init)

#5梯度下降
def gradientDescent(X,y,theta,alpha,iters,lamda):
    costs = []
    for i in range(iters):
        predictY = sigmoid(X @ theta)
        #正则化需要+上reg
        reg = theta[1:]*(lamda/len(X))
        reg=np.insert(reg,0,values=0,axis=0)

        theta = theta - (X.T @ (predictY - y)) * alpha / len(X) - reg
        cost = costFunction(X, y, theta, lamda)
        costs.append(cost)
        
        if i%1000 == 0:
            print(i,cost)
    return theta, costs

alpha = 0.001
iters = 200000
lamda = 0.001
theta_final, costs = gradientDescent(X,y,theta,alpha,iters,lamda)

#6 决策边界
x = np.linspace(-1.2,1.2,200)
#制造网格
xx,yy = np.meshgrid(x,x)
z = feature_mapping(xx.ravel(),yy.ravel(),6).values
zz = z @ theta_final
print('xx shape',xx.shape)
zz = zz.reshape(xx.shape)
print('zz shape',zz.shape)

fig,ax = plt.subplots()
ax.scatter(data[data['Accecpted']==0]['test1'],data[data['Accecpted']==0]['test2'],c='r',marker='x',label='y=0')
ax.scatter(data[data['Accecpted']==1]['test1'],data[data['Accecpted']==1]['test2'],c='b',marker='o',label='y=1')
ax.legend()
ax.set(
    xlabel='test1',
    ylabel='test2',
)
plt.contour(xx,yy,zz,0)
plt.show()

