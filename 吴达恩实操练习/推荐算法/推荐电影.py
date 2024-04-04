import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

#获取数据
mat = sio.loadmat('./ex8_movies.mat')
Y,R = mat['Y'],mat['R']
print(Y.shape,R.shape)
param_mat = sio.loadmat('./ex8_movieParams.mat')
X,Theta,nu,nm,nf = param_mat['X'],param_mat['Theta'],param_mat['num_users'],param_mat['num_movies'],param_mat['num_features']
print(X.shape,Theta.shape,nu,nm,nf)
nu = int(nu)
nm = int(nm)
nf = int(nf)
print(X.shape,Theta.shape,nu,nm,nf)

#序列化以及解序列化，为了后面minimize
def serialize(X,Theta):
    
    return np.append(X.flatten(),Theta.flatten())

def deserialize(params,nm,nu,nf):
    X = params[:nm*nf].reshape(nm,nf)
    Theta = params[nm*nf:].reshape(nu,nf)
    return X,Theta

#代价函数
def costFunction(params, Y, R,nm, nu, nf, lamda):
    X, Theta = deserialize(params,nm,nu,nf)
    error = 0.5*np.square((X@Theta.T-Y)*R).sum()
    reg1 = 0.5*lamda*np.square(X).sum()
    reg2 = 0.5*lamda*np.square(Theta).sum()
    return error + reg1 + reg2

# users = 4
# movies = 5
# features = 3
# X_sub = X[:movies,:features]
# Theta_sub = Theta[:users,:features]
# Y_sub = Y[:movies,:users]
# R_sub = R[:movies,:users]
# cost1 = costFunction(serialize(X_sub,Theta_sub),Y_sub,R_sub,movies,users,features,lamda = 0)
# print(cost1)

#梯度
def costGradient(params, Y, R,nm, nu, nf, lamda):
    X, Theta = deserialize(params,nm,nu,nf)
    X_grad = (((X@Theta.T)-Y)*R)@Theta + lamda * X
    Theta_grad = (((X@Theta.T)-Y)*R).T@X + lamda * Theta
    return serialize(X_grad,Theta_grad)
# def costGradient(params,Y,R,nm,nu,nf,lamda):
#     X,Theta = deserialize(params,nm,nu,nf)
#     X_grad = ((X@Theta.T-Y)*R)@Theta +lamda * X
#     Theta_grad = ((X@Theta.T-Y)*R).T@X + lamda * Theta
#     return serialize(X_grad,Theta_grad)

#新添加一个用户
my_ratings = np.zeros((nm,1))
my_ratings[9]   = 5
my_ratings[66]  = 5
my_ratings[96]   = 5
my_ratings[121]  = 4
my_ratings[148]  = 4
my_ratings[285]  = 3
my_ratings[490]  = 4
my_ratings[599]  = 4
my_ratings[643] = 4
my_ratings[958] = 5
my_ratings[1117] = 3

Y = np.c_[Y,my_ratings]
R = np.c_[R,my_ratings!=0]
nm,nu = Y.shape

#均值归一化
def normalizeRatings(Y, R):
    Y_mean = (Y.sum(axis=1) / R.sum(axis=1)).reshape(-1,1)
    print(2222,Y_mean.shape)
    Y_norm = (Y - Y_mean)*R
    return Y_norm,Y_mean
Y_norm, Y_mean = normalizeRatings(Y, R)
# def normalizeRatings(Y,R):
#     Y_mean =(Y.sum(axis=1) / R.sum(axis=1)).reshape(-1,1)
#     Y_norm = (Y - Y_mean) * R
#     return Y_norm,Y_mean
# Y_norm,Y_mean = normalizeRatings(Y,R)

#参数初始化
X = np.random.random((nm,nf))
Theta = np.random.random((nu,nf))
params = serialize(X,Theta)
lamda = 10

#模型训练
from scipy.optimize import minimize
res = minimize(
    fun = costFunction,
    x0=params,
    args = (Y_norm, R, nm, nu,nf,lamda),
    method = 'TNC',
    jac = costGradient,
    options= {'maxiter':100}
)
params_fit = res.x
fit_X, fit_Theta = deserialize(params_fit,nm,nu,nf)

#预测
Y_pred = fit_X @ fit_Theta.T
y_pred = Y_pred[:,-1] + Y_mean.flatten()
#降序排列
# index = np.argsort(-y_pred)
print(1111,index[:10])


