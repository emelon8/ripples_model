# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 09:36:56 2016

@author: eric
"""
from time import time
start_time = time()
import numpy as np
from scipy import randn
import scipy.signal as sg
import matplotlib.pyplot as plt
# my Python functions
import lif_neuron as lif
import chat_neuron as chatn
from Inoise import intrinsic_noise
from se import se_poisson

# simulation time
simulation_time = 500 # [ms]
dt = 0.05 # [ms]

# set number of neurons of each type
num_pyr = 2 # Pyramidal neurons
num_inh = 2 # Inhibitory neurons
num_chat = 1 # MS-DB ChAT neurons

# set connection probabilities
pyr_pyr_pr = 0
pyr_inh_pr = 0.2
pyr_chat_pr = 0
inh_inh_pr = 0.2
inh_pyr_pr = 0.2
inh_chat_pr = 0
chat_chat_pr = 0
chat_pyr_pr = 0.2
chat_inh_pr = 0.2

# generate connectivity matrix
con_matrix = np.zeros((num_pyr+num_inh+num_chat,num_pyr+num_inh+num_chat),dtype=bool)
for k in range(num_pyr):
    for n in range(num_pyr):
        con_matrix[k,n] = np.random.rand()<pyr_pyr_pr
    for n in np.arange(num_pyr,num_pyr+num_inh):
        con_matrix[k,n] = np.random.rand()<pyr_inh_pr
#    for n in np.arange(num_pyr+num_inh,num_pyr+num_inh+num_chat):
#        con_matrix[k,n] = np.random.rand()<pyr_chat_pr
    con_matrix[k,k] = False # Comment out if autapses are possible
for k in np.arange(num_pyr,num_pyr+num_inh):
    for n in np.arange(num_pyr,num_pyr+num_inh):
        con_matrix[k,n] = np.random.rand()<inh_inh_pr
    for n in range(num_pyr):
        con_matrix[k,n] = np.random.rand()<inh_pyr_pr
#    for n in np.arange(num_pyr+num_inh,num_pyr+num_inh+num_chat):
#        con_matrix[k,n] = np.random.rand()<inh_chat_pr
    con_matrix[k,k] = False # Comment out if autapses are possible
#for k in np.arange(num_pyr+num_inh,num_pyr+num_inh+num_chat):
#    for n in np.arange(num_pyr+num_inh,num_pyr+num_inh+num_chat):
#        con_matrix[k,n] = np.random.rand()<chat_chat_pr
#    for n in range(num_pyr):
#        con_matrix[k,n] = np.random.rand()<chat_pyr_pr
#    for n in np.arange(num_pyr,num_pyr+num_inh):
#        con_matrix[k,n] = np.random.rand()<chat_inh_pr
#    con_matrix[k,k] = False # Comment out if autapses are possible

# pyramidal neuron LIF properties
taump = 20 # pyramidal neuron membrane time constant [ms]
Rp = 40 # pyramidal neuron membrane capacitance [pF]
Vrestp = -70 # resting membrane potential [mV]
Vtp = -52 # threshhold value [mV]
Vrp = -59 # -65 reset value [mV]
Vpeak = 50 # peak voltage of spike [mV]
arpp = 2 # pyramidal neuron absolute refractory period [ms]
gampap = 0.19 # AMPA on pyramidal neurons [nS]
ggabap = 2.5 # GABA on pyramidal neurons [nS]
gnmdap = 0.06 # NMDA on pyramidal neurons [nS]
gampape = 0.25 # external AMPA on pyramidal neurons [nS]
vampap = 0 # AMPA reversal potential [mV]
vgabap = -70 # GABA reversal potential [mV]
vnmdap = 0 # NMDA reversal potential [mV]
Iepyrrate = 24e3 # Rate of Poisson synaptic input [Hz] / number of external synapses

# inhibitory neuron LIF properties
taumi = 10 # inhibitory neuron membrane time constant [ms]
Ri = 50 # pyramidal neuron membrane capacitance [pF]
Vresti = -70 # resting membrane potential [mV]
Vti = -52 # threshhold value [mV]
Vri = -59 # -65 reset value [mV]
arpi = 1 # inhibitory neuron absolute refractory period [ms]
gampai = 0.3 # AMPA on inhibitory neurons [nS]
ggabai = 4 # GABA on inhibitory neurons [nS]
gnmdai = 0.1 # NMDA on inhibitory neurons [nS]
gampaie = 0.4 # external AMPA on inhibitory neurons [nS]
vampai = 0 # AMPA reversal potential [mV]
vgabai = -70 # GABA reversal potential [mV]
vnmdai = 0 # NMDA reversal potential [mV]
Ieinhrate = 22e3 # Rate of Poisson synaptic input [Hz] / number of external synapses

## ChAT neuron eLIF properties
#Cc = 81.9 # 81.9 capacitance [pF]
#gLc = 1.3 # 3 base conductance for model, do not change
#ELc = -85 # reversal for EL
#EsiK = -93.1 # -85 reversal for IsiK
#slp = 10 # 10 slope factor or deltaT value for exponential term (0.1 is what papers commonly use: leak shifts f-I and doesn't do much to f-V)
#Vrc = -65 # -65 reset value
#Vtc = -59.5 # threshhold value
#taub = 152.7 # 180
#taubh = 11100 # 5000
#tauw = 125 # w time constant
#gBmax = 30.1 # 34.1 slow, potassium current maximum conductance
#gnoisec = 11 # 7 for leak, not really a conductance, just a multiplier
#filterfrequencyc = 100 # cutoff frequency [Hz]
#a = 0.1 # w parameters (conductance)
#B = 2.5 # w parameters (increment per spike)
#Iechatrate = 1e3 # Rate of Poisson synaptic input [Hz] / number of external synapses
#Vinitc = randn(num_chat)-75

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

# pyramidal neuron variables
# preallocate vectors
vp = np.zeros((len(t),num_pyr))
Ip = np.zeros((len(t),num_pyr))
spikesp = np.zeros((len(t),num_pyr)) # preallocate spike vectors for determining synaptic input and rate
arp_counterp = np.ones((len(t),num_pyr))
# set variable initial values
vp[0,:] = Vrestp
arp_counterp *= arpp / dt # set the counter equal to the arp so it doesn't reset the voltage

# inhibitory neuron variables
# preallocate vectors
vi = np.zeros((len(t),num_inh))
Ii = np.zeros((len(t),num_inh))
spikesi = np.zeros((len(t),num_inh))
arp_counteri = np.ones((len(t),num_inh))
# set variable initial values
vi[0,:] = Vresti
arp_counteri *= arpi / dt # set the counter equal to the arp so it doesn't reset the voltage

## ChAT neuron variables
## preallocate vectors
#vc = np.zeros((len(t),num_chat))
#IsiK = np.zeros((len(t),num_chat))
#w = np.zeros((len(t),num_chat))
#b = np.zeros((len(t),num_chat))
#bh = np.zeros((len(t),num_chat))
#spikesc = np.zeros((len(t),num_chat))
## set variable initial values
#vc[0,:] = Vinitc
#b[0,:] = 0.14+0.81/(1+np.exp((Vinitc+22.46)/-8.08)) # extracted from experiments
#bh[0,:] = 0.08+0.88/(1+np.exp((Vinitc+60.23)/5.69))
#
#Inoisechat = intrinsic_noise(t,dt,filterfrequencyc,gnoisec,num_chat) # Intrinsic noise in ChAT neurons

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

# generate Poisson synaptic input
sepyr = se_poisson(t,dt,Iepyrrate,num_pyr,sampakernelp/sampapeakp)
seinh = se_poisson(t,dt,Ieinhrate,num_inh,sampakerneli/sampapeaki)
#sechat = se_poisson(t,dt,Iechatrate,num_chat,sampakernelp/sampapeakp)

for i in range(len(t)-1):
    vp[i,vp[i,:]>Vtp] = Vpeak
    vi[i,vi[i,:]>Vti] = Vpeak
#    vc[i,(vc[i+1,:]==Vrc) & (vc[i,:]>Vtc)] = Vpeak

    # synaptic input based on last step's voltages from each population
    spikesp[i,vp[i,:]>0] = 1
    spikesi[i,vi[i,:]>0] = 1
#    spikesc[i+1,vc[i,:]>0] = 1

    # convolve spikes with the synaptic waveform to get the s-variable
    #    sampa = np.convolve(spikesp[:,k],sampakernelp)[0,len(t)]
    sampap = np.apply_along_axis(lambda m: sg.fftconvolve(m,sampakernelp),0,spikesp)[:len(t),:] # s-variables for inputs to pyramidal neurons
    sgabap = np.apply_along_axis(lambda m: sg.fftconvolve(m,sgabakernelp),0,spikesi)[:len(t),:]
    snmdap = np.apply_along_axis(lambda m: sg.fftconvolve(m,snmdakernelp),0,spikesp)[:len(t),:]
#    schatp =  np.apply_along_axis(lambda m: sg.fftconvolve(m,schatkernelp),0,spikesc)[:len(t),:]
    sampai = np.apply_along_axis(lambda m: sg.fftconvolve(m,sampakerneli),0,spikesp)[:len(t),:] # s-variables for inputs to inhibitory neurons
    sgabai = np.apply_along_axis(lambda m: sg.fftconvolve(m,sgabakerneli),0,spikesi)[:len(t),:]
    snmdai = np.apply_along_axis(lambda m: sg.fftconvolve(m,snmdakerneli),0,spikesp)[:len(t),:]
#    schati =  np.apply_along_axis(lambda m: sg.fftconvolve(m,schatkerneli),0,spikesc)[:len(t),:]

    vp[i+1,:], arp_counterp[i+1,:], Ip[i+1,:] = lif.LIF_neuron(dt,vp[i,:],taump,Rp,Vrestp,Vrp,Vtp,arpp,arp_counterp[i,:],sepyr[i,:],con_matrix[:,:num_pyr],num_pyr,num_inh,num_chat,sampap[i,:],sgabap[i,:],snmdap[i,:],gampap,ggabap,gnmdap,gampape,vampap,vgabap,vnmdap) # pyramidal neurons
    vi[i+1,:], arp_counteri[i+1,:], Ii[i+1,:] = lif.LIF_neuron(dt,vi[i,:],taumi,Ri,Vresti,Vri,Vti,arpi,arp_counteri[i,:],seinh[i,:],con_matrix[:,num_pyr:num_pyr+num_inh],num_pyr,num_inh,num_chat,sampai[i,:],sgabai[i,:],snmdai[i,:],gampai,ggabai,gnmdai,gampaie,vampai,vgabai,vnmdai) # inhibitory neurons
#    vc[i+1,:],IsiK_temp,w_temp,b_temp,bh_temp = chatn.chat_neuron(dt,vc[i,:],IsiK[i,:],w[i,:],b[i,:],bh[i,:],Inoisechat[i,:],dt,Cc,gLc,ELc,EsiK,slp,Vrc,Vtc,taub,taubh,tauw,gBmax,a,B,sechat[i,:],sampa[i,:],sgaba[i,:],snmda[i,:]) # ChAT neurons

    # Set next step as start of absolute refractory period
    arp_counterp[i+1,vp[i,:] == Vpeak] = 1
    arp_counteri[i+1,vi[i,:] == Vpeak] = 1

    if (t[i]==1000) | (t[i]==2000) | (t[i]==3000) | (t[i]==4000) | (t[i]==5000) | (t[i]==6000) | (t[i]==7000) | (t[i]==8000) | (t[i]==9000):
        print(time())


# find network rate for each type of neuron
prate = np.sum(spikesp,1)/dt*1000 # /dt*1000 to get rate in Hz
irate = np.sum(spikesi,1)/dt*1000
#vcrate = np.sum(vc[i,:]>0,0)

#np.histogram(bins=simulation_time/dt

plt.plot(t,prate+irate)
plt.xlabel('time [ms]')
plt.ylabel('network rate [spikes/s]')
plt.show()

total_time = time()-start_time #in seconds