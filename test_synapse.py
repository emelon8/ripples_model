# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:05:25 2016
 This script is used to confirm that the shape of the synapic conductance variable is correct.
@author: eric
"""

from time import time
start_time = time()
import numpy as np
import matplotlib.pyplot as plt

# simulation time
simulation_time = 1000 # [ms]
dt = 0.05 # [ms]

# set number of neurons of each type
num_pyr = 4000 # Pyramidal neurons
num_inh = 1000 # Inhibitory neurons
num_chat = 1 # MS-DB ChAT neurons

# pyramidal neuron LIF properties
taump = 20 # pyramidal neuron membrane time constant [ms]

# inhibitory neuron LIF properties
taumi = 10 # inhibitory neuron membrane time constant [ms]

# parameters for synaptic currents
taulampa = 1 # AMPA latency time constant [ms]
taulgaba = 0.5 # GABA latency time constant [ms]
taulnmda = 1 # NMDA latency time constant [ms]
taurampa = 0.4 # AMPA rise time constant [ms]
taurgaba = 0.5 # GABA rise time constant [ms]
taurnmda = 2 # NMDA rise time constant [ms]
taudampa = 2 # AMPA decay time constant [ms]
taudgaba = 5 # GABA decay time constant [ms]
taudnmda = 100 # NMDA decay time constant [ms]

# run the model
SR = (1/dt)*1000 #sample rate
total_time = simulation_time/1000 # total time to simulate [s]
t = np.arange(0,total_time*1000+dt,dt) # time array

vp = np.zeros((len(t),num_pyr))
vi = np.zeros((len(t),num_inh))

vp[10000,1]=1
vi[10000,1]=1

# synaptic variables
srampap = np.zeros(num_pyr)
sdampap = np.zeros(num_pyr)
srampai = np.zeros(num_pyr)
sdampai = np.zeros(num_pyr)
srgabap = np.zeros(num_inh)
sdgabap = np.zeros(num_inh)
srgabai = np.zeros(num_inh)
sdgabai = np.zeros(num_inh)
srnmdap = np.zeros(num_pyr)
sdnmdap = np.zeros(num_pyr)
srnmdai = np.zeros(num_pyr)
sdnmdai = np.zeros(num_pyr)

# synaptic kernels
sampapeakp = (taump/taudampa)*(taurampa/taudampa)**(taurampa/(taudampa-taurampa))
sampakernelp = (taump/(taudampa-taurampa))*(np.exp(-((t-taulampa)/taudampa))-np.exp(-((t-taulampa)/taurampa)))
sampakernelp[t<taulampa] = 0
sgabapeakp = (taump/taudgaba)*(taurgaba/taudgaba)**(taurgaba/(taudgaba-taurgaba))
sgabakernelp = (taump/(taudgaba-taurgaba))*(np.exp(-((t-taulgaba)/taudgaba))-np.exp(-((t-taulgaba)/taurgaba)))
sgabakernelp[t<taulgaba] = 0
snmdapeakp = (taump/taudnmda)*(taurnmda/taudnmda)**(taurnmda/(taudnmda-taurnmda))
snmdakernelp = (taump/(taudnmda-taurnmda))*(np.exp(-((t-taulnmda)/taudnmda))-np.exp(-((t-taulnmda)/taurnmda)))
snmdakernelp[t<taulnmda] = 0

sampapeaki = (taumi/taudampa)*(taurampa/taudampa)**(taurampa/(taudampa-taurampa))
sampakerneli = (taumi/(taudampa-taurampa))*(np.exp(-((t-taulampa)/taudampa))-np.exp(-((t-taulampa)/taurampa)))
sampakerneli[t<taulampa] = 0
sgabapeaki = (taumi/taudgaba)*(taurgaba/taudgaba)**(taurgaba/(taudgaba-taurgaba))
sgabakerneli = (taumi/(taudgaba-taurgaba))*(np.exp(-((t-taulgaba)/taudgaba))-np.exp(-((t-taulgaba)/taurgaba)))
sgabakerneli[t<taulgaba] = 0
snmdapeaki = (taumi/taudnmda)*(taurnmda/taudnmda)**(taurnmda/(taudnmda-taurnmda))
snmdakerneli = (taumi/(taudnmda-taurnmda))*(np.exp(-((t-taulnmda)/taudnmda))-np.exp(-((t-taulnmda)/taurnmda)))
snmdakerneli[t<taulnmda] = 0

# syntaptic vectors
sampap = np.zeros((len(t),num_pyr))
sgabap = np.zeros((len(t),num_inh))
snmdap = np.zeros((len(t),num_pyr))
sampai = np.zeros((len(t),num_pyr))
sgabai = np.zeros((len(t),num_inh))
snmdai = np.zeros((len(t),num_pyr))

for i in range(len(t)-1):

    if i >= taulampa/dt:
        srampap[vp[int(np.rint(i-taulampa/dt)),:]>0] += taump/(taudampa-taurampa)
        sdampap[vp[int(np.rint(i-taulampa/dt)),:]>0] += taump/(taudampa-taurampa)
        srampai[vp[int(np.rint(i-taulampa/dt)),:]>0] += taumi/(taudampa-taurampa)
        sdampai[vp[int(np.rint(i-taulampa/dt)),:]>0] += taumi/(taudampa-taurampa)
    if i >= taulgaba/dt:
        srgabap[vi[int(np.rint(i-taulgaba/dt)),:]>0] += taump/(taudgaba-taurgaba)
        sdgabap[vi[int(np.rint(i-taulgaba/dt)),:]>0] += taump/(taudgaba-taurgaba)
        srgabai[vi[int(np.rint(i-taulgaba/dt)),:]>0] += taumi/(taudgaba-taurgaba)
        sdgabai[vi[int(np.rint(i-taulgaba/dt)),:]>0] += taumi/(taudgaba-taurgaba)
    if i >= taulnmda/dt:
        srnmdap[vp[int(np.rint(i-taulnmda/dt)),:]>0] += taump/(taudnmda-taurnmda)
        sdnmdap[vp[int(np.rint(i-taulnmda/dt)),:]>0] += taump/(taudnmda-taurnmda)
        srnmdai[vp[int(np.rint(i-taulnmda/dt)),:]>0] += taumi/(taudnmda-taurnmda)
        sdnmdai[vp[int(np.rint(i-taulnmda/dt)),:]>0] += taumi/(taudnmda-taurnmda)
#    srp[i+1,vp[int(np.rint(i-taulampa/dt)),:]>0] = srp[i,vp[int(np.rint(i-taulampa/dt)),:]>0] + 1
#    sdp[i+1,vp[int(np.rint(i-taulampa/dt)),:]>0] = sdp[i,vp[int(np.rint(i-taulampa/dt)),:]>0] + 1
#    sri[i+1,vi[int(np.rint(i-taulampa/dt)),:]>0] = sri[i,vi[int(np.rint(i-taulampa/dt)),:]>0] + 1
#    sdi[i+1,vi[int(np.rint(i-taulampa/dt)),:]>0] = sdi[i,vi[int(np.rint(i-taulampa/dt)),:]>0] + 1

    srampap *= np.exp(-dt/taurampa)
    sdampap *= np.exp(-dt/taudampa)
    srampai *= np.exp(-dt/taurampa)
    sdampai *= np.exp(-dt/taudampa)

    srgabap *= np.exp(-dt/taurgaba)
    sdgabap *= np.exp(-dt/taudgaba)
    srgabai *= np.exp(-dt/taurgaba)
    sdgabai *= np.exp(-dt/taudgaba)

    srnmdap *= np.exp(-dt/taurnmda)
    sdnmdap *= np.exp(-dt/taudnmda)
    srnmdai *= np.exp(-dt/taurnmda)
    sdnmdai *= np.exp(-dt/taudnmda)

    sampap[i,:] = sdampap - srampap
    sampai[i,:] = sdampai - srampai
    sgabap[i,:] = sdgabap - srgabap
    sgabai[i,:] = sdgabai - srgabai
    snmdap[i,:] = sdnmdap - srnmdap
    snmdai[i,:] = sdnmdai - srnmdai