import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

#获取数据
data = sio.loadmat('./ex7faces.mat')
X = data['X']
print(X.shape)

# 图片显示
def plot_100_images(X):
    fig, axs = plt.subplots(ncols=10, nrows=10, figsize=(10,10))
    for c in range(10):
        for r in range(10):
            axs[c,r].imshow(X[10*c + r].reshape(32,32).T,cmap = 'Greys_r') #显示单通道的灰度图
            axs[c,r].set_xticks([])
            axs[c,r].set_yticks([])
# plot_100_images(X)
# plt.show()

#去均值
X_demean = X - np.mean(X, axis=0)
# plt.scatter(X_demean[:,0],X_demean[:,1])
# plt.show()

#计协方差矩阵
C = X_demean.T @ X_demean / len(X)
# print(C)

#计算特征值特征向量
U,S,V = np.linalg.svd(C)
U1 = U[:,:36]
X_reduction = X_demean @ U1
print(X_reduction.shape)

X_restore = X_reduction@U1.T + np.mean(X, axis=0)
plot_100_images(X_restore),plot_100_images(X)
plt.show()
