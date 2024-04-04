import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.svm import SVC

#获取数据
data = sio.loadmat('./data/ex6data1.mat')

X, y = data['X'], data['y']
print(X.shape,y.shape)
svc1 = SVC(C=1, kernel='linear')
svc1.fit(X,y.flatten())
print(svc1.predict(X))

def plot_boundary(model):
    x_min, x_max = -0.5, 4.5
    y_min, y_max = 1.3, 5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )
    print(111,xx,222,yy)
    z=model.predict(np.c_[xx.flatten(),yy.flatten()])
    zz = z.reshape(xx.shape)
    plt.contour(xx, yy, zz)

plot_boundary(svc1)
plt.show()
