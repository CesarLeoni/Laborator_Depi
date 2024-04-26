import numpy as np

def f (x,miu,sigma):
    return np.exp(-(x-miu)**2/(2*sigma**2))/((2*np.pi*sigma**2)**0.5)



##first exercise##
# x = np.load('files/lab5/ex1.npy')
# for valoare in x:
#     px0 = f(valoare,0,1)
#     px1 = f(valoare-1,0,1)
#     print(0 if px0 > px1 else 1)


## 2ND EXERCISE ##
# data_train = np.load('files/lab5/ex2_train.npy')
# x_train, y_train = data_train[:, 0], data_train[:, 1]
# data_test = np.load('files/lab5/ex2_test.npy')
# x_test, y_test = data_test[:, 0], data_test[:, 1]

# print(x_train.shape)
# print(x_test.shape)
#
# print(y_train)
# print(x_train)

# miu0 = np.mean(x_train[y_train==0])#media pentru x train atunci cand y train este 0
# miu1 = np.mean(x_train[y_train==1])
# sigma0 = np.std(x_train[y_train==0])
# sigma1 = np.std(x_train[y_train==1])
# y_pred = []
#
# for exemplu in x_test:
#     px0 = f(exemplu,miu0,sigma0)
#     px1 = f(exemplu,miu1,sigma1)
#     y_pred.append(0 if px0 > px1 else 1)
#
# y_pred = np.array(y_pred)
#
# print(np.sum(y_pred == y_test) / len(y_pred))#acuratete



##3RD EXERCISE##
# data_train = np.load('files/lab5/ex3_train.npy')
# x_train, y_train = data_train[:, :-1], data_train[:, -1]
# data_test = np.load('files/lab5/ex3_test.npy')
# x_test, y_test = data_test[:, :-1], data_test[:, -1]
#
# miu0 = np.mean(x_train[y_train==0],axis=0)#MEDIA LA NIVEL DE COLOANA axis=0,
# # #daca voiam pe linie veneea axis=1
# miu1 = np.mean(x_train[y_train==1],axis=0)
#
# sigma0 = np.std(x_train[y_train==0],axis=0)
# sigma1 = np.std(x_train[y_train==1],axis=0)
# y_pred = []
#
# for exemplu in x_test:
#     #px0 = np.prod(f(exemplu, miu0, sigma0))#numere foarte mici, varianta mai putin stabila
#     #px1 = np.prod(f(exemplu,miu1,sigma1))
#     px0 = np.sum(np.log(f(exemplu, miu0, sigma0)))#numere mai mari varianta stabila cu logaritmare
#     #nu vrem sa facem produsul si asa il evitam (ca sa nu avem nr f mici)
#     px1 = np.sum(np.log(f(exemplu, miu1, sigma1)))
#     print(px0,px1)
#     y_pred.append(0 if px0 > px1 else 1)
#
# y_pred = np.array(y_pred)
# print(np.sum(y_pred == y_test) / len(y_pred))



##4th EXERCISE##
data_train = np.load('files/lab5/ex4_train.npy')
x_train, y_train = data_train[:, :-1], data_train[:, -1]
data_test = np.load('files/lab5/ex4_test.npy')
x_test, y_test = data_test[:, :-1], data_test[:, -1]

miu = np.empty((int(np.max(y_train)) + 1, x_train.shape[1]))
sigma = np.empty((int(np.max(y_train)) + 1, x_train.shape[1]))

for i in range(int(np.max(y_train)) + 1):
    miu[i] = np.mean(x_train[y_train==i],axis=0)
    sigma[i] = np.std(x_train[y_train==i],axis=0)

y_pred = []
for exemplu in x_test:
    px = np.sum(np.log(f(exemplu,miu,sigma)),axis=1)
    y_pred.append(np.argmax(px))

y_pred = np.array(y_pred)
print(np.sum(y_pred == y_test) / len(y_pred))