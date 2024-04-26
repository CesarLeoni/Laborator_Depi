import numpy as np
def f (x,miu,sigma):
    return np.exp(-(x-miu)**2/(2*sigma**2))/((2*np.pi*sigma**2)**0.5)

x = np.load('files/lab5/ex1.npy')
for valoare in x:
    px0 = f(valoare,0,1)
    px1 = f(valoare-1,0,1)
    print(0 if px0 > px1 else 1)