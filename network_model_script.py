# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 09:36:56 2016

@author: eric
"""
from time import time
start_time = time()
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# my Python functions
import lif_neuron as lif
import chat_neuron as chatn
from Inoise import intrinsic_noise
from se import se_poisson

# simulation time
simulation_time = 1000 # [ms]
dt = 0.05 # [ms]

# set number of neurons of each type
num_pyr = 4000 # Pyramidal neurons
num_inh = 1000 # Inhibitory neurons
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
#    con_matrix[k,k] = False # Comment out if autapses are possible
for k in np.arange(num_pyr,num_pyr+num_inh):
    for n in np.arange(num_pyr,num_pyr+num_inh):
        con_matrix[k,n] = np.random.rand()<inh_inh_pr
    for n in range(num_pyr):
        con_matrix[k,n] = np.random.rand()<inh_pyr_pr
#    for n in np.arange(num_pyr+num_inh,num_pyr+num_inh+num_chat):
#        con_matrix[k,n] = np.random.rand()<inh_chat_pr
#    con_matrix[k,k] = False # Comment out if autapses are possible
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
gLp = 25 # pyramidal neuron leak conductance [nS] or resistance (40 MOhms)
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
gLi = 20 # pyramidal neuron leak conductance [nS] or resistance (50 MOhms)
Vresti = -70 # resting membrane potential [mV]
Vti = -52 # threshhold value [mV]
Vri = -59 # -65 reset value [mV]
arpi = 1 # inhibitory neuron absolute refractory period [ms]
gampai = 0.3 # AMPA on inhibitory neurons [nS]
ggabai = 4.0 # GABA on inhibitory neurons [nS]
gnmdai = 0.1 # NMDA on inhibitory neurons [nS]
gampaie = 0.4 #0.4 external AMPA on inhibitory neurons [nS]
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
#Iechatrate = 24e3 # Rate of Poisson synaptic input [Hz] / number of external synapses
#Vinitc = -75

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
#spikesp = np.zeros((len(t),num_pyr)) # preallocate spike vectors for determining synaptic input and rate
arp_counterp = np.ones((len(t),num_pyr))
# set variable initial values
vp[0,:] = Vrestp
arp_counterp *= arpp / dt # set the counter equal to the arp so it doesn't reset the voltage

# inhibitory neuron variables
# preallocate vectors
vi = np.zeros((len(t),num_inh))
#spikesi = np.zeros((len(t),num_inh))
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

# generate Poisson synaptic input
sepyr = se_poisson(t,dt,Iepyrrate,num_pyr,sampakernelp)
seinh = se_poisson(t,dt,Ieinhrate,num_inh,sampakerneli)
#sechat = se_poisson(t,dt,Iechatrate,num_chat,sampakernelp/sampapeakp)

DCp = 0 # offset current provided to pyramidal neurons [pA]
DCi = 0 # offset current provided to inhibitory neurons [pA]

prate = np.zeros(len(t))
irate = np.zeros(len(t))
#crate = np.zeros(len(t))

print('%s-DCi=%s,DCp=%s,Ieinhrate=%s,Iepyrrate=%s,SR=%s,Vpeak=%s,Vresti=%s,Vrestp=%s, \n \
      Vri=%s,Vrp=%s,Vti=%s,Vtp=%s,arpi=%s,arpp=%s,chat_chat_pr=%s,chat_inh_pr=%s,chat_pyr_pr=%s, \n \
      dt=%s,gLi=%s,gLp=%s,gampai=%s,gampaie=%s,gampap=%s,gampape=%s, \n \
      ggabai=%s,ggabap=%s,gnmdai=%s,gnmdap=%s,inh_chat_pr=%s,inh_inh_pr=%s,inh_pyr_pr=%s, \n \
      num_chat=%s,num_inh=%s,num_pyr=%s,pyr_chat_pr=%s,pyr_inh_pr=%s,pyr_pyr_pr=%s, \n \
      simulation_time=%s,taudampa=%s,taudgaba=%s,taudnmda=%s,taulampa=%s,taulgaba=%s,taulnmda=%s, \n \
      taumi=%s,taump=%s,taurampa=%s,taurgaba=%s,taurnmda=%s, \n \
      vampai=%s,vampap=%s,vgabai=%s,vgabap=%s,vnmdai=%s,vnmdap=%s' % \
      (start_time,DCi,DCp,Ieinhrate,Iepyrrate,SR,Vpeak,Vresti,Vrestp,Vri,Vrp,Vti,Vtp, \
      arpi,arpp,chat_chat_pr,chat_inh_pr,chat_pyr_pr,dt,gLi,gLp, \
      gampai,gampaie,gampap,gampape,ggabai,ggabap,gnmdai,gnmdap,inh_chat_pr,inh_inh_pr,inh_pyr_pr, \
      num_chat,num_inh,num_pyr,pyr_chat_pr,pyr_inh_pr,pyr_pyr_pr,simulation_time, \
      taudampa,taudgaba,taudnmda,taulampa,taulgaba,taulnmda,taumi,taump,taurampa,taurgaba,taurnmda, \
      vampai,vampap,vgabai,vgabap,vnmdai,vnmdap))

print('%s-you made it to the for loop at %s!' % (start_time, time()))

for i in range(len(t)-1):
    vp[i,vp[i,:]>Vtp] = Vpeak
    vi[i,vi[i,:]>Vti] = Vpeak
#    vc[i,vc[i,:]>Vtc] = Vpeak

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

    vp[i+1,:], arp_counterp[i+1,:] = lif.LIF_neuron(dt,vp[i,:],taump,gLp,Vrestp,Vrp,Vtp,arpp,arp_counterp[i,:],sepyr[i,:],con_matrix[:,:num_pyr],num_pyr,num_inh,num_chat,sampap[i,:],sgabap[i,:],snmdap[i,:],gampap,ggabap,gnmdap,gampape,vampap,vgabap,vnmdap,DCp) # pyramidal neurons
    vi[i+1,:], arp_counteri[i+1,:] = lif.LIF_neuron(dt,vi[i,:],taumi,gLi,Vresti,Vri,Vti,arpi,arp_counteri[i,:],seinh[i,:],con_matrix[:,num_pyr:num_pyr+num_inh],num_pyr,num_inh,num_chat,sampai[i,:],sgabai[i,:],snmdai[i,:],gampai,ggabai,gnmdai,gampaie,vampai,vgabai,vnmdai,DCi) # inhibitory neurons
#    vc[i+1,:],IsiK_temp,w_temp,b_temp,bh_temp = chatn.chat_neuron(dt,vc[i,:],IsiK[i,:],w[i,:],b[i,:],bh[i,:],Inoisechat[i,:],dt,Cc,gLc,ELc,EsiK,slp,Vrc,Vtc,taub,taubh,tauw,gBmax,a,B,sechat[i,:],sampa[i,:],sgaba[i,:],snmda[i,:]) # ChAT neurons

    # Set next step as start of absolute refractory period
    arp_counterp[i+1,vp[i,:] == Vpeak] = 1
    arp_counteri[i+1,vi[i,:] == Vpeak] = 1

    # find network rate for each type of neuron
    prate[i+1] = np.sum(vp[i,:]==Vpeak)*SR/num_pyr # to get rate in Hz/neuron
    irate[i+1] = np.sum(vi[i,:]==Vpeak)*SR/num_inh
#    crate[i+1] = np.sum(vc[i,:]==Vpeak)*SR/num_chat

    if i%(1000/dt) == 0:
        print('%s-you made it to the next second at %s!' % (start_time, time()))

## bin the network rate into time bins
#p_rate=np.zeros(10000)
#i_rate=np.zeros(10000)
#
#for ag in range(10000):
#    p_rate[ag]=sum(prate[int(ag/dt):int((ag+1)/dt)])
#    i_rate[ag]=sum(irate[int(ag/dt):int((ag+1)/dt)])

# smooth the network rate using a hamming window
bin_size = 1 # [ms]
i_rate_sum = pd.rolling_sum(irate/SR,bin_size/dt)*(1e3/bin_size)
p_rate_sum = pd.rolling_sum(prate/SR,bin_size/dt)*(1e3/bin_size)
i_rate = pd.rolling_window(irate,window=int(bin_size/dt),win_type='hamming')
p_rate = pd.rolling_window(prate,window=int(bin_size/dt),win_type='hamming')

# find and plot the FFT of the smoothed network rate
Li = np.size(i_rate)-sum(np.isnan(i_rate)) # 19982
Lp = np.size(p_rate)-sum(np.isnan(p_rate)) # 19982
ti = np.arange(0,Li+1)*dt
tp = np.arange(0,Lp+1)*dt
Youti = np.fft.fft(i_rate[sum(np.isnan(i_rate)):])
Youtp = np.fft.fft(p_rate[sum(np.isnan(p_rate)):])
P2i = np.abs(Youti/Li)
P2p = np.abs(Youtp/Lp)
P1i = P2i[:int((Li/2)+1)]
P1p = P2p[:int((Lp/2)+1)]
P1i[1:-1] = 2*P1i[1:-1]
P1p[1:-1] = 2*P1p[1:-1] # gets rid of DC and Nyquist; units are in [spikes*sqrt(Hz), or rate^2*Hz]
fi = SR*np.arange(0,int((Li/2)+1))/Li
fp = SR*np.arange(0,int((Lp/2)+1))/Lp

# plot the fraction of neurons at each rate
neuron_ratesp = sum(vp==Vpeak)/total_time
neuron_ratesi = sum(vi==Vpeak)/total_time
binned_neuron_ratesp = np.histogram(neuron_ratesp,bins=max(neuron_ratesp)*10)
binned_neuron_ratesi = np.histogram(neuron_ratesi,bins=max(neuron_ratesi)*10)
binned_fraction_neuron_ratesp = binned_neuron_ratesp[0]/num_pyr
binned_fraction_neuron_ratesi = binned_neuron_ratesi[0]/num_inh

computation_time = time()-start_time #in seconds

np.savez_compressed("saved_variables-%s" % start_time,DCi=DCi,DCp=DCp,Ieinhrate=Ieinhrate,Iepyrrate=Iepyrrate,SR=SR,Vpeak=Vpeak,Vresti=Vresti,
         Vrestp=Vrestp,Vri=Vri,Vrp=Vrp,Vti=Vti,Vtp=Vtp,arpi=arpi,arpp=arpp,chat_chat_pr=chat_chat_pr,chat_inh_pr=chat_inh_pr,
         chat_pyr_pr=chat_pyr_pr,computation_time=computation_time,con_matrix=con_matrix,dt=dt,gLi=gLi,gLp=gLp,gampai=gampai,
         gampaie=gampaie,gampap=gampap,gampape=gampape,ggabai=ggabai,ggabap=ggabap,gnmdai=gnmdai,gnmdap=gnmdap,
         inh_chat_pr=inh_chat_pr,inh_inh_pr=inh_inh_pr,inh_pyr_pr=inh_pyr_pr,irate=irate,neuron_ratesi=neuron_ratesi,
         neuron_ratesp=neuron_ratesp,num_chat=num_chat,num_inh=num_inh,num_pyr=num_pyr,prate=prate,pyr_chat_pr=pyr_chat_pr,
         pyr_inh_pr=pyr_inh_pr,pyr_pyr_pr=pyr_pyr_pr,simulation_time=simulation_time,t=t,taudampa=taudampa,taudgaba=taudgaba,
         taudnmda=taudnmda,taulampa=taulampa,taulgaba=taulgaba,taulnmda=taulnmda,taumi=taumi,taump=taump,taurampa=taurampa,
         taurgaba=taurgaba,taurnmda=taurnmda,total_time=total_time,vampai=vampai,vampap=vampap,vgabai=vgabai,vgabap=vgabap,
         vi=vi,vnmdai=vnmdai,vnmdap=vnmdap,vp=vp)

#np.savez_compressed("saved_variables_all",DCi=DCi,DCp=DCp,Ieinhrate=Ieinhrate,Iepyrrate=Iepyrrate,SR=SR,Vpeak=Vpeak,Vresti=Vresti,
#         Vrestp=Vrestp,Vri=Vri,Vrp=Vrp,Vti=Vti,Vtp=Vtp,arp_counteri=arp_counteri,arp_counterp=arp_counterp,arpi=arpi,arpp=arpp,
#         chat_chat_pr=chat_chat_pr,chat_inh_pr=chat_inh_pr,chat_pyr_pr=chat_pyr_pr,computation_time=computation_time,con_matrix=con_matrix,
#         dt=dt,gLi=gLi,gLp=gLp,gampai=gampai,gampaie=gampaie,gampap=gampap,gampape=gampape,ggabai=ggabai,ggabap=ggabap,gnmdai=gnmdai,gnmdap=gnmdap,
#         inh_chat_pr=inh_chat_pr,inh_inh_pr=inh_inh_pr,inh_pyr_pr=inh_pyr_pr,irate=irate,
#         neuron_ratesi=neuron_ratesi,neuron_ratesp=neuron_ratesp,num_chat=num_chat,num_inh=num_inh,num_pyr=num_pyr,prate=prate,
#         pyr_chat_pr=pyr_chat_pr,pyr_inh_pr=pyr_inh_pr,pyr_pyr_pr=pyr_pyr_pr,sampai=sampai,sampakerneli=sampakerneli,sampakernelp=sampakernelp,
#         sampap=sampap,sampapeaki=sampapeaki,sampapeakp=sampapeakp,sdampai=sdampai,sdampap=sdampap,sdgabai=sdgabai,sdgabap=sdgabap,
#         sdnmdai=sdnmdai,sdnmdap=sdnmdap,seinh=seinh,sepyr=sepyr,sgabai=sgabai,sgabakerneli=sgabakerneli,
#         sgabakernelp=sgabakernelp,sgabap=sgabap,sgabapeaki=sgabapeaki,sgabapeakp=sgabapeakp,simulation_time=simulation_time,
#         snmdai=snmdai,snmdakerneli=snmdakerneli,snmdakernelp=snmdakernelp,snmdap=snmdap,snmdapeaki=snmdapeaki,snmdapeakp=snmdapeakp,
#         srampai=srampai,srampap=srampap,srgabai=srgabai,srgabap=srgabap,srnmdai=srnmdai,srnmdap=srnmdap,t=t,taudampa=taudampa,
#         taudgaba=taudgaba,taudnmda=taudnmda,taulampa=taulampa,taulgaba=taulgaba,taulnmda=taulnmda,taumi=taumi,taump=taump,taurampa=taurampa,
#         taurgaba=taurgaba,taurnmda=taurnmda,total_time=total_time,vampai=vampai,vampap=vampap,vgabai=vgabai,vgabap=vgabap,vi=vi,vnmdai=vnmdai,
#         vnmdap=vnmdap,vp=vp)

# plot the raw network rates per dt
plt.figure()
plt.plot(t,irate)
plt.plot(t,prate,'r')
plt.xlabel('time [ms]')
plt.ylabel('network rate [spikes/s]')
plt.savefig('raw_rate-%s.svg' % start_time)

#plt.figure()
#plt.plot(np.linspace(0,10000),i_rate*dt)
#plt.xlabel('time [ms]')
#plt.ylabel('network rate [spikes/s]')
#plt.savefig('binned_rate-%s.svg' % start_time)

plt.figure()
plt.plot(t[14000:16000],i_rate_sum[14000:16000])
plt.plot(t[14000:16000],p_rate_sum[14000:16000],'r')
plt.xlabel('time [ms]')
plt.ylabel('network rate [spikes/s]')
plt.savefig('network_rate-%s.svg' % start_time)
plt.figure()
plt.plot(t[14000:16000],i_rate[14000:16000])
plt.plot(t[14000:16000],p_rate[14000:16000],'r')
plt.xlabel('time [ms]')
plt.ylabel('network rate [spikes/s]')
plt.savefig('network_rate_hamming-%s.svg' % start_time)

plt.figure()
plt.plot(fi,P1i)
plt.plot(fp,P1p,'r')
plt.xlabel('frequency [Hz]')
plt.ylabel('power')
plt.savefig('network_rate_fft-%s.svg' % start_time)
#plt.semilogx(f,P1)
#if max(P1i[1:])>=max(P1p[1:]):
#    plt.axis([0,1000,0,np.round(max(P1i[1:]),decimals=-2)+max(P1i[1:])*0.1])
#else:
#    plt.axis([0,1000,0,np.round(max(P1p[1:]),decimals=-2)+max(P1p[1:])*0.1])

## use the built in magnitude_spectrum function to find the energy at different frequencies
#plt.figure()
#magnitudespectrumi=plt.magnitude_spectrum(i_rate[19:],Fs=1e3/dt)
#magnitudespectrump=plt.magnitude_spectrum(p_rate[19:],Fs=1e3/dt,color='r')
#if max(magnitudespectrumi[1][20:])>=max(magnitudespectrump[1][20:]):
#    plt.axis([0,1000,0,5e6])#np.round(max(magnitudespectrumi[1][20:]),decimals=-2)+100])
#else:
#    plt.axis([0,1000,0,5e6])#np.round(max(magnitudespectrump[1][20:]),decimals=-2)+100])

plt.figure()
plt.bar(binned_neuron_ratesp[1][:-1]+np.mean(binned_neuron_ratesp[1][0:1]),binned_fraction_neuron_ratesp,color='r')
plt.xlabel('firing rate [spikes/s]')
plt.ylabel('fraction of neurons')
plt.savefig('neuron_rates_pyr-%s.svg' % start_time)
plt.figure()
plt.bar(binned_neuron_ratesi[1][:-1]+np.mean(binned_neuron_ratesi[1][0:1]),binned_fraction_neuron_ratesi)
plt.xlabel('firing rate [spikes/s]')
plt.ylabel('fraction of neurons')
plt.savefig('neuron_rates_inh-%s.svg' % start_time)