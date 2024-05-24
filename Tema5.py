############# 1

import numpy as np
def p1(a, r):
    p_r_S0 = 1 / (2 * np.pi * a ** 2) ** 0.5 * np.exp(-r ** 2 / (2 * a ** 2))
    p_r_S1 = 1 / (2 * np.pi * a ** 2) ** 0.5 * np.exp(-(r - 1) ** 2 / (2 * a ** 2))
    return p_r_S0, p_r_S1

############# 2

import numpy as np

def p2(X, C, R):
    media = np.zeros((2, X.shape[1]))
    deviatia = np.zeros((2, X.shape[1]))
    for i in range(2):
        media[i] = np.mean(X[C == i], axis=0)
        deviatia[i] = np.std(X[C == i], axis=0)
    f = 1 / ((2 * np.pi * deviatia[0] ** 2) ** 0.5) * np.exp((-(R-media[0]) ** 2) / (2 * deviatia[0] ** 2))
    f1 = np.prod(f, axis=1)
    g = 1 / ((2 * np.pi * deviatia[1] ** 2) ** 0.5) * np.exp((-(R-media[1]) ** 2) / (2 * deviatia[1] ** 2))
    g1 = np.prod(g, axis=1)
    rez = np.where(f1 > g1, 0, 1)
    return rez