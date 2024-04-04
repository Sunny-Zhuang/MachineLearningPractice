import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

#获取数据
data = sio.loadmat('./ex8data1.mat')
X = data['X']
Xval, yval = data['Xval'], data['yval']
# print(X.shape,7777,Xval[:10,:],yval[:10,:])

#计算均值和方差
def estimateGaussian(X, isCovariance):
    means = np.mean(X,axis=0)
    if isCovariance:
        sigma2 = (X-means).T@(X-means)/len(X)
    else: 
        sigma2 = np.var(X,axis=0)
    return means, sigma2

means, sigma2= estimateGaussian(X, isCovariance=True)
# print(1111,means,2222,sigma2)

# 多元正太分布的密度函数
def gaussian(X, means, sigma2):
    if np.ndim(sigma2) == 1:
        sigma2 = np.diag(sigma2)
     
    X = X - means
    n = X.shape[1]
    print(999,n)
    
    first = np.power(2*np.pi,-n/2)*(np.linalg.det(sigma2)**(-0.5))
    second =np.diag(X@np.linalg.inv(sigma2)@X.T) 
    p = first * np.exp(-0.5*second)
    p = p.reshape(-1,1)
    return p

#绘图
def plotGaussian(X, means, sigma2):
    x = np.arange(0,30,0.5)
    y = np.arange(0,30,0.5)
    xx, yy = np.meshgrid(x,y)
    z = gaussian(np.c_[xx.ravel(),yy.ravel()], means, sigma2)
    zz = z.reshape(xx.shape)
    plt.plot(X[:,0],X[:,1],'bx')
    contour_levels = [10**h for h in range(-20,0,3)]
    plt.contour(xx, yy, zz, contour_levels)
# plotGaussian(X, means, sigma2)
# plt.show()

#选取yuzhi
def selectThreshold(yval,p):
    bestF1 = 0
    bestEpsilon = 0
    epsilons = np.linspace(min(p),max(p),1000)
    for e in epsilons:
        p_ = p<e
        tp = np.sum((yval==1)&(p_==1))
        fp = np.sum((yval==0)&(p_==1))
        fn = np.sum((yval==1)&(p_==0))
        prec = tp / (tp+fp) if(tp+fp) else 0
        rec = tp / (tp+fn) if(tp+fn) else 0
        F1_e = 2*prec*rec/(prec+rec) if(prec+rec) else 0
        if F1_e > bestF1:
            bestF1 = F1_e
            bestEpsilon = e   

    return bestF1, bestEpsilon

means,sigma2 = estimateGaussian(X,isCovariance = True)
pval = gaussian(Xval,means,sigma2)
bestF1,bestEpsilon = selectThreshold(yval,pval)
print(4444,bestF1,5555,bestEpsilon)

p = gaussian(X, means, sigma2)
anoms = np.array([X[i] for i  in range(X.shape[0]) if p[i]<bestEpsilon])
plotGaussian(X, means, sigma2)
plt.scatter(anoms[:,0],anoms[:,1],c='red',marker='o')
plt.show()

