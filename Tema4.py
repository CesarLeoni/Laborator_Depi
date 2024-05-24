###########PROBLEMA 1
import numpy as np
def p1(seed_xi, seed_eta, a, b):
    f = 100
    fs = 2000
    t = np.arange(0, 50/1000, 1/fs)
    np.random.seed(seed_xi)
    xi = np.random.normal(0, a, size = 500)
    np.random.seed(seed_eta)
    eta = np.random.normal(0, b, size = 500)
    x = np.empty((500, len(t)))
    for i in range(0, 500):
        x[i] = xi[i] * np.cos(2 * np.pi * f * t) + eta[i] * np.sin(2 * np.pi * f *t)
    mean_vector = np.mean(x, axis=0)
    autocorrelation_matrix = np.zeros((len(t), len(t)))
    for i in range(len(t)):
        for j in range(len(t)):
            autocorrelation_matrix[i, j] = np.mean(x[:, i] * x[:, j])

    return x, mean_vector, autocorrelation_matrix


###########PROBLEMA 2

import numpy as np
from scipy import signal

def p2(filename):
    fs = 441e2
    data = np.loadtxt(filename)
    [f,PSD] = signal.periodogram(data, fs)
    freq_m = f[np.argmax(PSD)]
    return freq_m


###################PROBLEMA 3

def p3(f1, f0):
    A = 1
    T = 1
    fs = 500
    t = np.arange(0, T, 1/fs)
    theta = 2*np.pi*((f1-f0)/(2*T)*t**2 + f0*t)
    x = A*np.sin(theta)
    R = np.correlate(x, x, mode='full')
    [f,PSD] = signal.periodogram(x,fs ,nfft = 1024)
    PSD = 10*np.log10(PSD/np.max(PSD))
    for i in range (1,len(PSD)):
        if(PSD[i] <=-20 ):
            return f[i]

##########PROBLEMA 4
import numpy as np
def p4(f):
    A = 1
    T = 1
    fs = 1000
    t = np.arange(0, 1, 1 / fs)
    x = A * np.sin(2*np.pi*f*t)
    R = np.correlate(x, x, mode='full')
    index = []
    for i in range(1,len(R)-1):
        if(R[i]>R[i-1] and R[i]>R[i+1]):
            index.append(i)
    return index