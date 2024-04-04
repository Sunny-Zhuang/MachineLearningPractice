import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.optimize import minimize

# 一对多策略
# 一对多策略是一种常用的将二分类方法扩展到多分类问题的方法。假设我们有K个类别，对于每一个类别，我们都会训练一个二分类器。这个二分类器的目标是识别当前类别与其他所有类别的区别。

# 具体地说，对于第i个类别，我们训练一个二分类器，这个分类器会试图将第i个类别的样本与不是第i个类别的样本区分开。为了实现这一点，我们将原始的多类标签转换为二值标签。如果样本属于第i个类别，则新标签为1；如果样本不属于第i个类别，则新标签为0。

#获取数据
data = sio.loadmat('./ex3data1.mat')
# print(data)
raw_X = data['X']
raw_y = data['y']
# print(11111,raw_X.shape,2222,raw_y.shape)
# 打印出原始图像1张
# def plot_an_image(X):
#     pick_one = np.random.randint(5000)
#     image = X[pick_one,:]
#     fig,ax = plt.subplots(figsize=(1,1))
#     ax.imshow(image.reshape(20,20).T,cmap='gray_r')
#     #取消刻度
#     plt.xticks([])
#     plt.yticks([])

# plot_an_image(raw_X)
# plt.show()

# 打印出原始图像100张
# def plot_an_image(X):
#     sample_index = np.random.choice(len(X),100)
#     images = X[sample_index,:]
#     fig,ax = plt.subplots(ncols=10,nrows=10,figsize=(8,8),sharex=True,sharey=True)

#     for r in np.arange(10):
#         for c in np.arange(10):
#             ax[r,c].imshow(images[10*r+c].reshape(20,20).T,cmap='gray_r')
#     #取消刻度
#     plt.xticks([])
#     plt.yticks([])

# plot_an_image(raw_X)
# plt.show()

#5 损失函数
def sigmoid(z): 
    return 1 / (1 + np.exp(-z))

def costFunction(theta, X, y, lamda):
    predictY = sigmoid(X @ theta)
    #注意是点乘，不是矩阵乘法
    first = y * np.log(predictY)
    second = (1-y) * np.log(1-predictY)

    # l2正则化，由于特征映射可能会导致过拟合所以需要正则化
    # reg = np.sum(np.power(theta[1:],2))*(lamda/(2*len(X)))
    reg = theta[1:]@theta[1:]*(lamda/(2*len(X)))

    #注意累加 需要用np.sum
    return -np.sum(first + second) / len(y) + reg

# theta = np.zeros((28,1))
# lamda = 1
# cost_init = costFunction(X, y, theta, lamda)
# print('cost_init', cost_init)

#5梯度下降
# 当你使用 minimize 函数并指定 method='TNC' 时，minimize 会尝试使用截断牛顿法来找到目标函数的最小值。为了实现这一点，它需要知道目标函数的梯度。这就是 gradient_reg 函数的作用所在。你可以将 gradient_reg 作为 minimize 函数的 jac 参数（代表雅可比矩阵，即梯度）传入，以便 TNC 方法能够使用梯度信息来更有效地进行优化。
def gradientDescent(X,y,theta,alpha,iters,lamda):
    costs = []
    for i in range(iters):
        predictY = sigmoid(X @ theta)
        #正则化需要+上reg
        reg = theta[1:]*(lamda/len(X))
        reg = np.insert(reg,0,values=0,axis=0)

        theta = theta - (X.T @ (predictY - y)) * alpha / len(X) - reg
        cost = costFunction(X, y, theta, lamda)
        costs.append(cost)
        
        if i%1000 == 0:
            print(i,cost)
    return theta, costs

def gradient_reg(theta, X, y, lamda):
    reg = theta[1:]*(lamda/len(X))
    reg = np.insert(reg,0,values=0,axis=0)
    first = (X.T @ (sigmoid(X @ theta) - y)) / len(X)
    return first + reg

X = np.insert(raw_X,0,values=1,axis=1)
print('x',X.shape)

print('raw y',raw_y.shape)
y = raw_y.flatten()
print('y',y.shape)

#6 使用scipy.optimize 算法得到多分类的theta
def one_vs_all(X, y, lamda, K):
    n = X.shape[1]
    theta_all = np.zeros((K,n))

    for i in range(1,K+1):
        theta_i = np.zeros(n,)

        res = minimize(
            fun = costFunction,
            x0 = theta_i,
            args = (X, y == i, lamda),
            method = 'TNC',
            jac = gradient_reg
        )
        theta_all[i-1,:] = res.x
    return theta_all

lamda  = 1
K = 10
theta_final = one_vs_all(X, y, lamda, K)
print(theta_final)

# 预测
def predic(X, theta_final):
    h = sigmoid(X @ theta_final.T)
    h_argmax = np.argmax(h, axis=1)
    return h_argmax+1

y_pred = predic(X, theta_final)
acc = np.mean(y_pred==y)
print('acc',acc)

