# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 14:35:10 2016

@author: eric
"""
from time import time
start_time=time()
import numpy as np
import scipy.signal as sg

def se_poisson(t,dt,Ierate,num_neurons,synaptic_kernel): # convolve a Poisson process with synaptic input
    synaptic_T = se = np.zeros((len(t),num_neurons))
    for k in range(num_neurons):
        last_sT = np.array([((-1000/Ierate)/dt)*np.log(np.random.random())]) # find first event [index]
        #last_sT = np.array([np.random.poisson((1000/Ierate)/dt)]) # THIS DIDN'T WORK
        while last_sT[-1]<len(t)-1:
            last_sT = np.append(last_sT,last_sT[-1]+((-1000/Ierate)/dt)*np.log(np.random.random())); # find next event [index]
            #last_sT = np.append(last_sT,last_sT[-1]+np.random.poisson((1000/Ierate)/dt)) # THIS DIDN'T WORK
#            last_sT = len(synaptic_T[:,k])-np.argmax(synaptic_T[::-1,k])-1 # alternative way to find the index of the last event if you assign synaptic_T along the way
        synaptic_T[np.rint(last_sT[:-1]).astype(np.int),k] = 1
        se[:,k] = sg.fftconvolve(synaptic_T[:,k],synaptic_kernel)[:len(t)] # np.convolve() is much slower
    return se

## In case you want to test the function:
#t = np.linspace(0,10000,200001)
#taump = 20
#ampataul = 1
#ampataur = 0.5
#ampataud = 2
#speak = (taump/ampataud)*(ampataur/ampataud)**(ampataur/(ampataud-ampataur))
#s = (taump/(ampataud-ampataur))*(np.exp(-((t-ampataul)/ampataud))-np.exp(-((t-ampataul)/ampataur)))
#s[t<ampataul] = 0
#cool = se_poisson(t,0.05,10,500,s)
#
#print((time()-start_time)/60)
#
##import matplotlib.pyplot as ply
##
##ply.plot(t,cool)
##ply.xlabel('time [ms]')
##ply.ylabel('gating variable')
##ply.show()