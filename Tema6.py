#############problema 1
import numpy as np

def p1(miu_u, sigma_u, R):
    theta_map = (np.sum(R) / 0.1 + miu_u / (sigma_u**2)) / (len(R) / 0.1 + 1 / (sigma_u**2))
    return theta_map

############problema 2
import numpy as np
def p2(R):
    sigma2 = 0.1
    U = np.mean(R)
    rmax = np.max(R)
    r = np.arange(0, rmax, 0.01)
    P = (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-(r - U) ** 2 / (2 * sigma2))
    maxim = np.argmax(P)
    return r[maxim]