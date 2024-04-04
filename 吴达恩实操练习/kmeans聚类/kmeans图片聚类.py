import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
# from skimage import io

#获取数据
data = sio.loadmat('./bird_small.mat')
A = data['A']
print(A.shape)

A = A/255
A = A.reshape(-1,3)
print(A.shape)

def find_centroids(X, centros):
    idx = []
    for i in range(len(X)):
        dist = np.linalg.norm((X[i]-centros), axis = 1)
        id_i = np.argmin(dist)
        idx.append(id_i)
    return np.array(idx)
# centros = np.array([[3,3],[6,2],[8,5]])
# idx = find_centroids(X, centros)
# print(idx)

#计算聚类中心点
def compute_centros(X, idx, k):
    centros = []
    for i in range(k):
        centros_i = np.mean(X[idx == i], axis=0)
        centros.append(centros_i)
    return np.array(centros)

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

#随机选择中心点
def init_centros(X,k):
    index = np.random.choice(len(X),k)
    return X[index]

k=16
idx, centros_all=run_kmeans(A, init_centros(A, 16), 20)
centros = centros_all[-1]
im = np.zeros(A.shape)
for i in range(k):
    im[idx==i] = centros[i]
im=im.reshape(128,128,3)
plt.imshow(im)
plt.show()

