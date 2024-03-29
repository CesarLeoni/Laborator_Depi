import numpy as np
import matplotlib.pyplot as plt

# data = np.load("penguins.npy")
#
# print(data)
# x = data[:, 0]
# y = data[:, 1]
# plt.scatter(x, y)
#
# R=np.mean(x*y)
# print(R)
#
# K=R-np.mean(x)*np.mean(y)
# print(K)
#
# ro=K/(np.std(x)*np.std(y))
# print(ro)
#
# plt.show()

from mpl_toolkits.mplot3d import Axes3D
pas= 0.01
X = np.arange(4,  7, pas)
Y = np.arange(1, 4, pas)
X, Y = np.meshgrid(X, Y)
miu = np.array([5.5, 2.5])
sigma = np.array([[1, 0.5], [0.5, 2]])
det = np.linalg.det(sigma)
inv = np.linalg.inv(sigma)
f = np.empty((len(X), len(Y)))
for i in range(len(X)):
    for j in range(len(Y)):
        C = np.array([X[i, j], Y[i, j]])
        f[i, j] = np.exp((-1/2) * (C-miu) @ inv @ (C-miu))/((2*np.pi)**(2/2)*det**0.5)# 3.14 (X = C) in formula 3.14 notat cu x iar in cod cu C
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, f)
plt.show()