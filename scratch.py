# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 11:41:37 2016

@author: eric
"""
#p_rate=np.zeros(100)
#i_rate=np.zeros(100)
#for i in range(100):
#    p_rate[i]=sum(prate[i*20:(i+1)*20])
#    i_rate[i]=sum(irate[i*20:(i+1)*20])
#
#plt.plot(np.arange(0,100)/10,p_rate,'r')
#plt.plot(np.arange(0,100)/10,i_rate)

#import numpy as np
#import matplotlib.pyplot as plt
#import scipy.fftpack
#
#Fs = 20000
#T = 1/Fs
#L = 19982
#t = np.arange(0,L+1)*T
#Yout = np.fft.fft(i_rate[19:])
#P2 = np.abs(Yout/L)
#P1 = P2[:int((L/2)+1)]
#P1[1:-1] = 2*P1[1:-1]
#f = Fs*np.arange(0,int((L/2)+1))/L
#
#plt.plot(f,P1)
#plt.semilogx(f,P1)
#plt.axis([0,1000,0,2500])

#plt.figure()
#magnitudespectrumi=plt.magnitude_spectrum(i_rate[19:],Fs=Fs)
#magnitudespectrump=plt.magnitude_spectrum(p_rate[19:],Fs=Fs)

#Fs = 20000;
#T = 1/Fs;
#L = 19982;
#t = (0:L)*T;
#Yout = fft(i_rate(20:end));
#P2 = abs(Yout/L);
#P1 = P2(1:(L/2)+1);
#P1(2:end-1) = 2*P1(2:end-1);
#f = Fs*(0:L/2)/L;
#
#plot(f,P1)


## Number of samplepoints
#N = 20000
## sample spacing
#T = 1.0 / 20000.0
#x = np.linspace(0.0, N*T, N)
#yf = scipy.fftpack.fft(i_rate[1:])
#xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
#
#fig, ax = plt.subplots()
#ax.plot(xf, 2.0/N * np.abs(yf[:N/2]))
#plt.show()
#
## Number of samplepoints
#N = 600
## sample spacing
#T = 1.0 / 800.0
#x = np.linspace(0.0, N*T, N)
#y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
#yf = scipy.fftpack.fft(y)
#xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
#
#fig, ax = plt.subplots()
#ax.plot(xf, 2.0/N * np.abs(yf[:N/2]))
#plt.show()

import numpy as np

tmp = file("C:\\Windows\\Temp\\temp_npz.npz",'wb')

## some variables
#a= [23,4,67,7]
#b= ['w','ww','wwww']
#c= np.ones((2,6))

# a lit containing the name of your variables
var_list=dir()['a','b','c']

# save the npz file with the variables you selected
str_exec_save = "np.savez(tmp,"    
for i in range(len(var_list)):    
    str_exec_save += "%s = %s," % (var_list[i],var_list[i])
str_exec_save += ")"
exec(str_exec_save)

tmp.close