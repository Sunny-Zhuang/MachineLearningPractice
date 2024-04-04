import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.optimize import minimize

#获取数据
data = sio.loadmat('./ex5data1.mat')
#训练集
X_train = data['X']
y_train = data['y']
X_train = np.insert(X_train,0,values=1,axis=1)
print(X_train.shape,y_train.shape)
#验证集
X_val = data['Xval']
y_val = data['yval']
X_val = np.insert(X_val,0,values=1,axis=1)
print(X_val.shape,X_val.shape)
#测试集
X_test = data['Xtest']
y_test = data['ytest']
X_test = np.insert(X_test,0,values=1,axis=1)
print(X_test.shape,y_test.shape)


#损失函数
def reg_cost(theta,X,y,lamda):
    inner = np.power(X @ theta -y.flatten(),2)
    cost = np.sum(inner)
    reg = theta[1:]@theta[1:] * lamda
    return (cost + reg) / (2 * len(X))

lamda = 1
theta = np.ones(X_train.shape[1])
print(theta)
print(reg_cost(theta,X_train,y_train,lamda))

#下降函数
def gradient_reg(theta, X, y, lamda):
    grad = ((X @ theta) - y.flatten()) @ X
    reg = lamda * theta
    reg[0] = 0
    return (reg+grad)/len(X)

print(gradient_reg(theta,X_train,y_train,lamda))

def train_model(X,y,lamda):
    res = minimize(
        fun = reg_cost,
        x0=theta,
        args=(X,y,lamda),
        method='TNC',
        jac = gradient_reg
    )
    return res.x

print(train_model(X_train,y_train,lamda=0))
theta_final = train_model(X_train,y_train,lamda=0)

# x = np.linspace(-40,40,12)
# y_ = X_train@theta_final


def plot_data():
    fig,ax = plt.subplots()
    ax.scatter(X_train[:,1],y_train,label="training data")
    # ax.plot(X_train[:,1],y_,'r',label="predict")

# fig,ax = plt.subplots()
# ax.scatter(X_train[:,1],y_train,label="training data")
# ax.plot(X_train[:,1],y_,'r',label="predict")
# # scattor和plot同时显示
# ax.legend()
# ax.set(
#     xlabel='population',
#     ylabel='profit',
# )
# plt.show()

#根据样本的数量，查看测试集和验证集
def plot_learning_curve(X_train,y_train,X_val,y_val,lamda):
    x=range(1,len(X_train)+1)
    training_cost =[]
    cv_cost = []

    for i in x:
        res = train_model(X_train[:i,:],y_train[:i,:],lamda)
        training_cost_i = reg_cost(res,X_train[:i,:],y_train[:i,:],lamda)
        training_cost.append(training_cost_i)
        cv_cost_i = reg_cost(res,X_val,y_val,lamda)
        cv_cost.append(cv_cost_i)

    fig,ax = plt.subplots()
    # ax.scatter(X_train[:,1],y_train,label="training data")
    ax.plot(x,training_cost,'r',label="training cost")
    ax.plot(x,cv_cost,'b',label="cv cost")
    # scattor和plot同时显示
    ax.legend()
    plt.show()

# plot_learning_curve(X_train,y_train,X_val,y_val,lamda=0)
# 从图中看cv和traingcost都比较大，都高于真实值，高偏差是欠拟合的，需要调整模型
# 如果是高方差就需要加test数据，就是都跟真实值差不多，但两条线离得比较远
# 基本jtrain高，jcv高是欠拟合，jtrain低，jcv高是过拟合

#根据以上分析 构造更多维度的多项式,并归一化
def poly_feature(X,power):
    for i in range(2,power+1):
        X = np.insert(X,X.shape[1],np.power(X[:,1],i),axis=1)
    return X

def get_means_stds(X):
    means = np.mean(X,axis=0)
    stds = np.std(X,axis=0)
    return means, stds

def feature_normalize(X,means,stds):
    X[:,1:] = (X[:,1:] - means[1:])/stds[1:]
    return X

power = 6
X_train_poly = poly_feature(X_train, power)
X_val_poly = poly_feature(X_val, power)
X_test_poly = poly_feature(X_test, power)

train_means,train_stds = get_means_stds(X_train)

X_tain_norm = feature_normalize(X_train,train_means,train_stds)
X_val_norm = feature_normalize(X_val,train_means,train_stds)
X_test_norm = feature_normalize(X_test,train_means,train_stds)

theta_fit = train_model(X_tain_norm, y_train,lamda=0)

# plot_learning_curve(X_tain_norm,y_train,X_val_norm,y_val,lamda=1)

# 取合适的lamda
lamdas = [0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]

train_cost = []
cv_cost = []
for lamda in lamdas:
    res = train_model(X_tain_norm, y_train,lamda)
    tc = reg_cost(res, X_tain_norm, y_train, 0)
    cv = reg_cost(res, X_val_norm, y_val, 0)
    train_cost.append(tc)
    cv_cost.append(cv)

plt.plot(lamdas,train_cost,label='training cost')
plt.plot(lamdas,cv_cost,label='cv cost')
plt.legend()

plt.show()






    

