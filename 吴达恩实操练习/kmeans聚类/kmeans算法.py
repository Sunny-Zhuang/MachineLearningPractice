import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

#获取数据
data = sio.loadmat('./ex7data2.mat')
X = data['X']
print(X.shape)

#matplot
# plt.scatter(X[:,0],X[:,1])
# plt.show()

# 遍历所有x，将数据划分到最近的那个类中心点
#计算点X[i]与一组中心点centros之间的欧几里得距离，并找出距离最小的中心点的索引。
def find_centroids(X, centros):
    idx = []
    for i in range(len(X)):
        dist = np.linalg.norm((X[i]-centros), axis = 1)
        id_i = np.argmin(dist)
        idx.append(id_i)
    return np.array(idx)
centros = np.array([[3,3],[6,2],[8,5]])
idx = find_centroids(X, centros)
# print(idx)

#计算聚类中心点
def compute_centros(X, idx, k):
    centros = []
    for i in range(k):
        centros_i = np.mean(X[idx == i], axis=0)
        centros.append(centros_i)
    return np.array(centros)

print(compute_centros(X, idx, 3))

#运行kmeans执行2，3
def run_kmeans(X, centros, iters):
    k = len(centros)
    centros_all = []
    centros_all.append(centros)
    centros_i = centros
    for i in range(iters):
        idx = find_centroids(X, centros_i)
        centros_i = compute_centros(X, idx, k)
        centros_all.append(centros_i)
    return idx, np.array(centros_all)

#绘制数据集和聚类中心的移动轨迹
def plot_data(X, centros_all, idx):
    plt.figure()
    plt.scatter(X[:,0],X[:,1],c=idx,cmap='rainbow')
    plt.plot(centros_all[:,:,0],centros_all[:,:,1],'kx--')
    plt.show()

idx, centros_all=run_kmeans(X, centros, 10)
plot_data(X, centros_all, idx)



