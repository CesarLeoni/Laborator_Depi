#SUBIECTUL 1
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# plt.figure()
# fs=2000
# f=100
# t = np.arange(0,0.2,1/fs)#era (0,2PI,0.1) inainte
# X = np.empty((500, len(t)))
#
# for i in range(500):
#     beta = np.random.uniform(-np.pi,np.pi)
#     alfa = np.random.normal(0,1)
#     wn=np.random.normal(0,1,len(t))#zgomot alb, orice in fizica cauzeaa zgomot alb
#     #X[i] = alfa* np.cos(2*np.pi*f*t)+beta*np.sin(4*np.pi*f*t)#era cos(t+alfa)
#     #X[i] = alfa * np.sin(2 * np.pi * f * t + beta)
#     X[i] = np.sin(2 * np.pi * f * t)+wn
#     if i % 100 == 0:
#         plt.plot(t, X[i])
#
#
# plt.plot(t,np.mean(X,axis=0) , linestyle='dashed')
#
# R = np.empty((len(t), len(t)))
# for t1 in range(len(t)):
#     for t2 in range(len(t)):
#         R[t1, t2] = np.mean(X[:,t1]*X[:,t2])#inmulteste element cu
#         #element coloanele de la t1 i t2
# plt.figure()
# plt.imshow(R)
#
# plt.show()
# # X= cos este SSL - are benzi in matricea de corelatie
# # pt -1,1 nu mai e SSL - media nu mai e constanta, e sinus,si nu are benzi matricea de corelatie
#
#
# #SUBIECTUL 2
# #SSL are acelasi raspuns in frecventa
# #FFT = transformata fourier
#
#
# R_SSL = R[0, : ]
# S = np.abs(np.fft.fft(R_SSL)[:len(R_SSL) // 2 + 1])
# freq = np.linspace(0, 0.5, len(S)) * fs
# plt.figure()
# plt.plot(freq, S)
# f_PSD, PSD = signal.periodogram(X[0], fs)
# f_PSD1, PSD1 = signal.periodogram(X[1], fs)
# plt.figure()
# plt.plot(f_PSD, PSD)
# plt.plot(f_PSD1, PSD1)
# print(freq[np.argmax(S)], f_PSD[np.argmax(PSD)], f_PSD1[np.argmax(PSD1)] )
# plt.show()

#Problema 4
import librosa #pentru wav
#pentru fisierele npy fs trebuie initializat de noi

x,fs=librosa.load("files/AAAA.wav")#inregistrat de mn
y,fs=librosa.load("files/2.wav")
z,fs=librosa.load("files/3.wav")
print(fs)
print(x.shape)

fpsdx, psdx = signal.periodogram(x,fs)
fpsdy, psdy = signal.periodogram(y,fs)
fpsdz, psdz = signal.periodogram(z,fs)



plt.figure()
plt.plot(fpsdx,psdx)
plt.xlim(0,1000)
plt.figure()
plt.plot(fpsdy,psdy)
plt.xlim(0,1000)
plt.figure()
plt.plot(fpsdz,psdz)
plt.xlim(0,1000)# ne arata de la plotul dinainte doar pana la x 1000
plt.show()

#PROBLEMA 5
f = 2
fs = 2000
t = np.arange(0,2,1/fs)
a=1
b=1
c=1
X = np.sin(2 * np.pi * f * (a * t ** 2 + b * t + c))

plt.figure()
plt.plot(t,X)
plt.figure()
fpsd, psd = signal.periodogram(X,fs)
plt.plot(fpsd,psd)
plt.xlim(0,50)
plt.show()

