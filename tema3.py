import numpy as np
def p1(numefisier):
    with open(numefisier, 'r') as f:
        mama=f.readlines()
        xstr = mama[0].split(' ')
        ystr = mama[1].split(' ')

        xx = [float(val) for val in xstr]
        yy = [float(val) for val in ystr]


    x = np.array(xx)
    y = np.array(yy)
    N = len(x)

    # Calculul coeficienților regresiei liniare
    mean_X = np.mean(x)
    mean_Y = np.mean(y)
    SS_xy = np.sum(x * y) - N * mean_X * mean_Y
    SS_xx = np.sum(x * x) - N * mean_X * mean_X
    b1 = SS_xy / SS_xx
    b0 = mean_Y - b1 * mean_X

    return b1, b0
    pass


B1,B0 = p1("text_tema_3.txt")

print(f"B0 este {B0}")
print(f"B1 este {B1}")



from scipy.stats import multivariate_normal

# Definirea matricei de covarianță
cov_matrix = np.array([[1, 0.2, 0.2],
                       [0.2, 1, 0.2],
                       [0.2, 0.2, 1]])

# Definirea functiei p2(x)
def p2(x):
    mean = np.zeros_like(x)  # Mediile sunt toate zero
    mvn = multivariate_normal(mean=mean, cov=cov_matrix)
    return mvn.pdf(x)

# Exemplu de utilizare
x = np.array([0, 0, 0])  # Punctul în care calculăm densitatea de probabilitate
density = p2(x)
print(f"Densitatea de probabilitate în punctul {x} este: {density}")