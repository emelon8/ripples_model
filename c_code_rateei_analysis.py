# -*- coding: utf-8 -*-
"""
Created on Thu May 12 18:20:14 2016
Data analysis for text file generated with C code
@author: eric
"""
import numpy as np
import matplotlib.pyplot as plt

#t=voltagetxt[:,0]
#v=voltagetxt[:,1]-70
#v[1:-2][np.diff(np.asarray(np.diff(v)==0, dtype=int))>0]=50
#plt.plot(t,v)

dt=0.001
SR=1000
t2=rateeitxt[:,0]
ratei=rateeitxt[:,2] # spikes/neuron/s
ratee=rateeitxt[:,1]/4 #*1000/4000 => /4 to get in spikes/neuron/s

#plt.figure()
#plt.plot(t2,ratei,'k')
#plt.plot(t2,ratee,'r')
#plt.xlabel('time [ms]')
#plt.ylabel('network rate [spikes/neuron/s]')
#plt.axis([1000,1100,0,100])

plt.figure()
plt.fill_between(t2,ratei,color='k')
plt.fill_between(t2,ratee,color='r')
plt.xlabel('time [ms]')
plt.ylabel('network rate [spikes/neuron/s]')
plt.axis([1000,1100,0,1000])

# find and plot the FFT of the network rate
Li = np.size(ratei) # 19982
Lp = np.size(ratee) # 19982
ti = np.arange(0,Li+1)*dt
tp = np.arange(0,Lp+1)*dt
Youti = np.fft.fft(ratei)
Youtp = np.fft.fft(ratee)
P2i = np.abs(Youti/Li)
P2p = np.abs(Youtp/Lp)
P1i = P2i[:int((Li/2)+1)]
P1p = P2p[:int((Lp/2)+1)]
P1i[1:-1] = 2*P1i[1:-1]
P1p[1:-1] = 2*P1p[1:-1] # gets rid of DC and Nyquist; units are in [(spikes/s)/sqrt(Hz), or (spikes/s)^2/Hz]
fi = SR*np.arange(0,int((Li/2)+1))/Li
fp = SR*np.arange(0,int((Lp/2)+1))/Lp

plt.figure()
plt.plot(fi,P1i**2)
plt.plot(fp,P1p**2,'r')
plt.xlabel('frequency [Hz]')
plt.ylabel('power [(spikes/s)^2/Hz]')
#plt.axis([0,500,0,.4e-25])
plt.xlim([0,500])

# combined rates
ratetot=ratei+ratee;

Ltot = np.size(ratetot) # 19982
ttot = np.arange(0,Ltot+1)*dt
Youttot = np.fft.fft(ratetot)
P2tot = np.abs(Youttot/Ltot)
P1tot = P2tot[:int((Ltot/2)+1)]
P1tot[1:-1] = 2*P1tot[1:-1]
ftot = SR*np.arange(0,int((Ltot/2)+1))/Ltot

plt.figure()
plt.plot(ftot,P1tot**2)
plt.xlabel('frequency [Hz]')
plt.ylabel('power [(spikes/s)^2/Hz]')
#plt.axis([0,500,0,.4e-25])
plt.xlim([0,500])