# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:08:45 2016

@author: eric
"""
import numpy as np
import matplotlib.pyplot as plt
import ripplefunctions as rp

#for k in range(100):
#    plt.figure()
#    plt.fill_between(np.arange(.05,70+0.05,.05),mean_recoveryv[k,:]-sem_recoveryv[k,:],mean_recoveryv[k,:]+sem_recoveryv[k,:],color='k',alpha=0.5)
#    plt.plot(np.arange(.05,70+0.05,.05),mean_recoveryv[k,:],'k')
#    plt.show()
#    input('Press Enter to continue...')

ra,rt,rf=rp.recoveryrunningavg(mean_all_recoveryv,windowsz=108,dt=0.05)