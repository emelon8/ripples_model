# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 14:59:57 2016

@author: eric
"""

import numpy as np
from scipy import randn, sqrt

def intrinsic_noise(t,dt,filterfrequency,gnoise,num_neurons=1): # noise generation for any number of neurons
#    Inoise = np.zeros((len(t),num_neurons))
    dt_ins = dt/1000
    df = 1/(t[-1]/1000+dt_ins) # freq resolution
    fidx = np.tile(np.arange(1,len(t)/2,1),(num_neurons,1)).T # np.arange(1,len(t)/2+1,1) it has to be N/2 pts, where N=len(t); Python makes a range from 1 to np.ceil(len(t)/2)-1
    faxis = (fidx-1)*df
    #make the phases
    Rr = randn(len(t)/2,num_neurons) # randn(np.size(fidx)) # ~N(0,1) over [-1,1]
    distribphases = np.exp(1j*np.pi*Rr) # on the unit circle
    #make the amplitudes - filtered
    filterf = sqrt(1/((2*np.pi*filterfrequency)**2+(2*np.pi*faxis)**2))

    fourierA = distribphases*filterf # representation in fourier domain
    # make it conj-symmetric so the ifft is real
    fourierB = fourierA.conj()[::-1]
    nss = np.concatenate((np.tile([0],(1,num_neurons)),fourierA,fourierB))
    Inoise = np.fft.ifft(nss,axis=0)
    scaling = np.std(Inoise,axis=0,ddof=1)
    Inoise = Inoise/scaling
    Inoise = Inoise*gnoise
    return Inoise