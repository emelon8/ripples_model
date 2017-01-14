# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 14:35:10 2016

@author: eric
"""

import numpy as np
import scipy.signal as sg

def se_poisson(t,dt,Ierate,num_neurons,synaptic_kernel): # convolve a Poisson process with synaptic input
    synaptic_T = se = np.random.poisson(Ierate*(dt/1e3),(len(t),num_neurons))
    for k in range(num_neurons):
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
#cool = se_poisson(t,0.05,24e3,500,s)
#
##import matplotlib.pyplot as plt
##
##plt.plot(t,cool)
##plt.xlabel('time [ms]')
##plt.ylabel('gating variable')
##plt.show()