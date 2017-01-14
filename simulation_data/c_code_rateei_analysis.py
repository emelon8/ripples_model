# -*- coding: utf-8 -*-
"""
Created on Thu May 12 18:20:14 2016
Data analysis for text file generated with C code
@author: eric
"""
import numpy as np
import matplotlib.pyplot as plt
import ripplefunctions as rp

plt.close('all')

rateeitxt=np.loadtxt('rateei.txt',delimiter=',')

dt=0.001
SR=1000
t2=rateeitxt[:,0]
ratei=rateeitxt[:,2] # spikes/neuron/s
ratee=rateeitxt[:,1]/4 #*1000/4000 => /4 to get in spikes/neuron/s

plt.figure()
plt.subplot(1,3,1)

plt.fill_between(t2,ratei,color='b')
plt.fill_between(t2,ratee,color='r')
plt.xlabel('time [ms]')
plt.ylabel('network rate [spikes/neuron/s]')
plt.axis([1000,1100,0,100])
#        
# find and plot the FFT of the network rate
fi, P1i2, _ = rp.ripplepower(ratei,dt)
fp, P1p2, _ = rp.ripplepower(ratee,dt)
ratetot=ratei+ratee;
ftot, P1tot2, _ = rp.ripplepower(ratetot,dt)

plt.subplot(1,3,2)
plt.plot(fi,P1i2,'b')
plt.plot(fp,P1p2,'r')
plt.xlabel('frequency [Hz]')
plt.ylabel('power [(spikes/s)^2/Hz]')
plt.axis([0,500,0,40])#np.max(np.array(np.max(P1i[40:-1]**2),np.max(P1p[40:-1]**2)))*1.1])

plt.subplot(1,3,3)
plt.plot(ftot,P1tot2,'k')
plt.xlabel('frequency [Hz]')
plt.ylabel('power [(spikes/s)^2/Hz]')
plt.axis([0,500,0,40])

rippleintegral,extrarippleintegral,totalpower,maxpowerfreq,ripples=rp.powerintegrals(P1tot2, ftot)

rp.ripplespectrogram(ratee)
rp.ripplespectrogram(ratei)
rp.ripplespectrogram(ratetot)

ripple_synchony=rp.sts(ratetot,ratetot)

mean_all_spikes, sem_all_spikes, cell_num_spikes, cell_mean_spike, cell_sem_spike, raster_times = rp.STA('voltage_1-100.txt')
rp.ripplesraster(raster_times)
n,binedges=rp.ripplesrates(cell_num_spikes)