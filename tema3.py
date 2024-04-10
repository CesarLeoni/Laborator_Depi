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