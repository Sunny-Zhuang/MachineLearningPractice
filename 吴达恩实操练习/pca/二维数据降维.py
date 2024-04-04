import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

#获取数据
data = sio.loadmat('./ex7data1.mat')
X = data['X']
print(X.shape)

#去均值
X_demean = X - np.mean(X, axis=0)
# plt.scatter(X_demean[:,0],X_demean[:,1])
# plt.show()

#计协方差矩阵
C = X_demean.T @ X_demean / len(X)
print(C)

#计算特征值特征向量
U,S,V = np.linalg.svd(C)
U1 = U[:,0]
U2 = U[:,1]

#实现降维
X_reduction = X_demean @ U1

# plt.figure(figsize=[7,7])
# plt.scatter(X_demean[:,0],X_demean[:,1])
# plt.plot([0,U1[0]],[0,U1[1]],c='r')
# plt.plot([0,U2[0]],[0,U2[1]],c='k')
# plt.show()

#数据还原
X_restore = X_reduction.reshape(50,1)@U1.reshape(1,2) + np.mean(X, axis=0)
plt.scatter(X[:,0],X[:,1])
plt.scatter(X_restore[:,0],X_restore[:,1])
plt.show()