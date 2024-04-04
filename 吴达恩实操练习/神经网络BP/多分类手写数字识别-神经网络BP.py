import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


#获取数据
data = sio.loadmat('./ex4data1.mat')
# print(data)
raw_X = data['X']
raw_y = data['y']
print(111,raw_y.shape)

X = np.insert(raw_X,0,values=1,axis=1)
print(X.shape)

#对y进行独热编码处理 one-hot编码
def one_hot_encoder(raw_y):
    result = []
    for i in raw_y:
        y_temp = np.zeros(10)
        y_temp[i - 1] = 1
        result.append(y_temp)
    return np.array(result)
y = one_hot_encoder(raw_y)
print(one_hot_encoder(raw_y).shape)

# #获取theta权重参数
data = sio.loadmat('./ex4weights.mat')
theta1 = data['Theta1']
theta2 = data['Theta2']
print(theta1.shape)
print(theta2.shape)
#序列化权重参数
def serialize(a,b):
    return np.append(a.flatten(),b.flatten())

theta_serialize = serialize(theta1,theta2)
print(theta_serialize.shape)

#解序列化权重参数
def deserialize(theta_serialize):
    theta1 = theta_serialize[:25*401].reshape(25,401)
    theta2 = theta_serialize[25*401:].reshape(10,26)
    return theta1, theta2
theta1,theta2 = deserialize(theta_serialize)
print(theta1.shape,theta2.shape)

# # 开始向前传播
def sigmoid(z): 
    return 1 / (1 + np.exp(-z))

def feed_forward(theta_serialize, X):
    theta1, theta2 = deserialize(theta_serialize)
    # print('feee1',theta1,'feee2',theta2)
    a1 = X
    z2 = X @ theta1.T
    a2 = sigmoid(z2)
    #加入偏执项
    a2 = np.insert(a2,0,values=1,axis=1)
    # print('a2', a2.shape)

    z3 = a2 @ theta2.T
    h = sigmoid(z3)
    return a1,z2,a2,z3,h

# 损失函数
# 不带正则化
def cost(theta_serialize, X, y):
    a1,z2,a2,z3,h = feed_forward(theta_serialize, X)
    J = -np.sum(y * np.log(h) + (1-y)*np.log(1-h)) / len(X)
    return J

print(cost(theta_serialize, X, y))

# 带正则化
def reg_cost(theta_serialize, X, y, lamda):
    sum1 = np.sum(np.power(theta1[:,1:],2))
    sum2 = np.sum(np.power(theta2[:,1:],2))
    reg = (sum1 + sum2)*(lamda/(2*len(X)))
    # 逻辑回归线性不可分做对比
    # reg = np.sum(np.power(theta[1:],2))*(lamda/(2*len(X)))
    return reg + cost(theta_serialize, X, y)

lamda = 1
cost_init = reg_cost(theta_serialize, X, y, lamda)
print('cost_init', cost_init)

# 梯度 反向传播
# 逻辑回归线性不可分做对比
# theta = theta - (X.T @ (predictY - y)) * alpha / len(X) - reg
def sigmoid_gradient(z):
    return sigmoid(z)*(1-sigmoid(z))

def gradient(theta_serialize,X,y):
    theta1,theta2 = deserialize(theta_serialize)
    a1,z2,a2,z3,h = feed_forward(theta_serialize, X)
    d3 = h-y
    d2 = d3 @ theta2[:,1:] * sigmoid_gradient(z2)
    D2 = (d3.T @ a2)/len(X)
    D1 = (d2.T @ a1)/len(X)
    return serialize(D1,D2)
#带正则
def reg_gradient(theta_serialize,X,y,lamda):
    D = gradient(theta_serialize,X,y)
    D1, D2 = deserialize(D)
    theta1,theta2 = deserialize(theta_serialize)
    D1[:,1:] = D1[:,1:] + theta1[:,1:]*lamda/len(X)
    D2[:,1:] = D2[:,1:] + theta2[:,1:]*lamda/len(X)
    return serialize(D1,D2)

#神经网络优化
from scipy.optimize import minimize

def nn_training(X, y):
    init_theta = np.random.uniform(-0.5,0.5,10285)
    res = minimize(
        fun = reg_cost,
        x0 = init_theta,
        args = (X, y, lamda),
        method = 'TNC',
        jac = reg_gradient,
        options = {'maxiter': 300}
    )
    return res
lamda = 10
res = nn_training(X, y)

raw_y = raw_y.reshape(5000,)

# #预测
# print('res',res)
_,_,_,_,h = feed_forward(res.x, X)
pred = np.argmax(h,axis=1)
pred = pred + 1
acc = np.mean(pred == raw_y)
print(acc)

# 可视化隐藏层
def plot_hidden_layer(theta):
    theta1, _ = deserialize(theta)
    hidden_layer = theta1[:,1:]

    fig,ax = plt.subplots(ncols=5,nrows=5,figsize=(8,8),sharex=True,sharey=True)
    for r in np.arange(5):
        for c in np.arange(5):
            ax[r,c].imshow(hidden_layer[5*r+c].reshape(20,20).T,cmap='gray_r')
    #取消刻度
    plt.xticks([])
    plt.yticks([])

    # plot_an_image(raw_X)
    plt.show()
plot_hidden_layer(res.x)