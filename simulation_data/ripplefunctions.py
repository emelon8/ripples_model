# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:01:45 2016

@author: eric
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.stats import sem
from scipy.signal import spectrogram, hilbert
from scipy import random
from pandas import rolling_mean


# Power spectrum
def ripplepower(signal,dt=0.001):
    SR=1/dt
    L = np.size(signal)
    signal = np.reshape(signal,(L,))
    t = np.arange(0,L+1)*dt
    Yout = np.fft.fft(signal-np.mean(signal))
    P2 = np.abs(Yout/L)
    P1 = P2[:int((L/2)+1)] # fft returns a mirror image, so cut it in half
    P1[1:-1] = 2*P1[1:-1] # since you cut off half of the mirror image, double it? Also get rid of DC and Nyquist
    f = SR*np.arange(0,int((L/2)+1))/L
    P12 = P1**2
    return f, P12, t


def powerintegrals(P12, f):
    ripples=1
    maxpower=np.max(P12[40:3000])
    totalpower=np.sum(P12)
    maxpowerindex=np.where(P12[40:3000]==maxpower)[0][0]
    maxpowerfreq=maxpowerindex*(f[1]-f[0])
    rippleintegral=np.sum(P12[40:3000][maxpowerindex-400:maxpowerindex+400])
#    rippleintegral=np.sum((P12)[1400:2201])
    extrarippleintegral=np.sum((P12)[40:500])#maxpowerindex-400])
#    extrarippleintegral=np.sum((P12)[np.r_[40:1400,2201:len(P12)]])
#    centerofmass=np.sum((P12)[40:3000]*f[40:3000])/np.sum((P12)[40:3000])
#    if 140 <= centerofmass and centerofmass <= 220:
#        ripples=0
    if 140 <= maxpowerfreq and maxpowerfreq <= 220:
        ripples=0
    return rippleintegral, extrarippleintegral, totalpower, maxpowerfreq, ripples


# Spectrogram
def ripplespectrogram(s,samplefreq=1000,windowsize=500,xlab='time [s]',ylab='frequency [Hz]',cbarlab='power [(spikes/s)^2/Hz]'):
    f,t,Sxx=spectrogram(s,fs=samplefreq,nperseg=windowsize)
    plt.figure()
    plt.pcolormesh(t,f,Sxx)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    cbar=plt.colorbar()
    cbar.ax.set_ylabel(cbarlab)


# Spike Train Synchrony Index
def sts(rate1,rate2,normalized=True):
    if normalized:
        return np.correlate(rate1,rate2)/(np.mean(rate1)**2)
    else:
        return np.correlate(rate1,rate2)


# Surface plot of integral of power
def ripplessurf(X,Y,ripplepower,totalpower,normalized=True,titlelab='normalized SPW-R-ratio [140-220 Hz power/total power] integral',xlab='DCi',ylab='DCe'):
    rippleratio=ripplepower/totalpower
    fig = plt.figure(num=random.random_integers(100,200))
    ax = fig.gca(projection='3d')
    if normalized:
        surf = ax.plot_surface(X, Y, rippleratio/rippleratio[0,0], rstride=1, cstride=1,cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
    else:
        surf = ax.plot_surface(X, Y, rippleratio, rstride=1, cstride=1,cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(titlelab)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()


# Import variable that updates every timestep
def importvariable(filenames,num_cells=100,sim_time=10000,dt=0.05):
    # After loading the data from the text files as "variable_temp", prepare
    # the data and find all the spikes. If there are 500 cells in each of the
    # files and the simulation time is 10000 ms in 0.05 ms steps, the first 500
    # lines are the variable at t=0.05 ms for cells 1-500, the next 500 lines
    # are t=0.1 ms for cells 1-500, etc.
    sim_steps=sim_time/dt #timesteps
    variable_temp=np.loadtxt(filenames,delimiter=',',usecols=range(1))
    variable = np.zeros((num_cells,int(sim_steps)))
    for k in range(num_cells):
        variable[k,:]=variable_temp[np.r_[0:np.size(variable_temp):num_cells]+k]
    return variable


# Calculate recovery time of an IPSP
def recoverytimesteps(v,dt=0.05,tbeforespk=20,tafterspk=50,depthsearcht=10,refractoryv=-59,recoveryfraction=0.632):
    v=np.reshape(v,((tbeforespk+tafterspk)/dt,))
    v2=v[tbeforespk/dt:]
    if depthsearcht>tafterspk:
        depthsearcht=tafterspk
    v2minv=v2[:depthsearcht/dt] # voltage from the start of the IPSP to depthsearcht
    minrecoveryv=np.min(v2minv[np.where(v2minv!=refractoryv)[0]]) # minimum voltage from IPSP start to depthsearcht, excluding any refractory voltages
    minrecoveryt=np.where(v2==minrecoveryv)[0][0]
    depthrecoveryv=v2[0]-minrecoveryv #np.mean(v[:tbeforespk/dt])-minrecoveryv
    if np.size(np.where(v2[minrecoveryt:]>=(minrecoveryv+recoveryfraction*depthrecoveryv))[0])==0 or depthrecoveryv<=0:
        recoverysteps=0 # GABAergic events where the voltage doesn't recover to 0.632*depth within the tafterspk are set to 0 and discarded later; so are cases where the voltage doesn't go lower than at the IPSP start during the depthsearcht
    else:
        recoverysteps=np.where(v2[minrecoveryt:]>=(minrecoveryv+recoveryfraction*depthrecoveryv))[0][0]+minrecoveryt#-tbeforespk/dt # <== adding this will give you the time including the before the spike section
    return recoverysteps


# Calculate the mean variable over a set amount of time for a population of cells
def meanvariable(filename,num_cells=100,sim_time=10000,dt=0.05,startt=0,endt=None,title='',xlab='time [ms]',ylab='current [pA]'):
    variable=importvariable(filename,num_cells=num_cells,sim_time=sim_time,dt=dt)
    plt.figure()
    if endt:
        mean_variable=np.mean(variable[int(startt/dt):endt/dt],axis=0)
        plt.plot(np.arange(dt,endt-startt+dt,dt),mean_variable)
    else:
        mean_variable=np.mean(variable[int(startt/dt):],axis=0)
        plt.plot(np.arange(dt,sim_time+dt,dt),mean_variable)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    return mean_variable


# Spike triggered average
def STA(filenameV,filenameI=None,num_cells=100,sim_time=10000,dt=0.05,reset=11,tbeforespk=25,tafterspk=10,xlab='time [ms]',ylab='membrane voltage [mV]'):
    sim_steps=sim_time/dt #timesteps
    v=importvariable(filenameV,num_cells=num_cells,sim_time=sim_time,dt=dt)-70
    if filenameI:
        I=importvariable(filenameI,num_cells=num_cells,sim_time=sim_time,dt=dt)
    raster_times=[]
    cell_num_spikes=np.zeros(num_cells)
    sta_raster_times=[]
    cell_num_sta_spikes=np.zeros(num_cells)
    cell_spikes=[]
    cell_spikesI=[]
    cell_mean_spike=np.zeros((num_cells,(tbeforespk+tafterspk)/dt))
    cell_sem_spike=np.zeros((num_cells,(tbeforespk+tafterspk)/dt))
    cell_mean_spikeI=np.zeros((num_cells,(tbeforespk+tafterspk)/dt))
    cell_sem_spikeI=np.zeros((num_cells,(tbeforespk+tafterspk)/dt))
    all_spikes=np.empty((1,(tbeforespk+tafterspk)/dt))
    all_spikesI=np.empty((1,(tbeforespk+tafterspk)/dt))
    for k in range(num_cells):
        v[k,np.append(np.array(np.diff(np.array(np.logical_and(np.diff(v[k,:])==0,v[k,1:]==reset-70),dtype=int))>0,dtype=bool),np.array([0,0],dtype=bool))]=50 # set all time steps before refractory periods equal to 50 mV
        raster_times.append(np.where(v[k,:]==50)[0]) # find all spikes
        cell_num_spikes[k]=np.size(raster_times[k])
        sta_raster_times.append(raster_times[k][np.logical_and(raster_times[k]>tbeforespk/dt-1,raster_times[k]<sim_steps-tafterspk/dt)]) # find only those spikes that will give a full STA (aren't at the very beginning or end of the time)
        cell_num_sta_spikes[k]=np.size(sta_raster_times[k])
        cell_spikes.append(np.zeros((cell_num_sta_spikes[k],(tbeforespk+tafterspk)/dt)))
        cell_spikesI.append(np.zeros((cell_num_sta_spikes[k],(tbeforespk+tafterspk)/dt)))
        if cell_num_sta_spikes[k]>0:
            for n in range(int(cell_num_sta_spikes[k])):
                cell_spikes[k][n,:]=v[k,sta_raster_times[k][n]-tbeforespk/dt:sta_raster_times[k][n]+tafterspk/dt]
                if filenameI:
                    cell_spikesI[k][n,:]=I[k,sta_raster_times[k][n]-tbeforespk/dt:sta_raster_times[k][n]+tafterspk/dt]
            all_spikes=np.concatenate((all_spikes,cell_spikes[k]),axis=0)
            if filenameI:
                all_spikesI=np.concatenate((all_spikesI,cell_spikesI[k]),axis=0)
            cell_mean_spike[k,:]=np.mean(cell_spikes[k],0)
            cell_sem_spike[k,:]=sem(cell_spikes[k],0)
            if filenameI:
                cell_mean_spikeI[k,:]=np.mean(cell_spikesI[k],0)
                cell_sem_spikeI[k,:]=sem(cell_spikesI[k],0)
#            # find and plot the average spikes for each cell
#            plt.figure()
#            plt.fill_between(np.arange(dt,tbeforespk+tafterspk+dt,dt),cell_mean_spike[k,:]-cell_sem_spike[k,:],cell_mean_spike[k,:]+cell_sem_spike[k,:],color='k',alpha=0.5)
#            plt.plot(np.arange(dt,tbeforespk+tafterspk+dt,dt),cell_mean_spike[k,:],'k')
#            plt.xlabel(xlab)
#            plt.ylabel(ylab)
    sorted_indices=np.argsort(cell_num_spikes)
    cell_num_spikes.sort()
    raster_times=[raster_times[i] for i in sorted_indices]
    cell_mean_spike=cell_mean_spike[sorted_indices,:]
    cell_sem_spike=cell_sem_spike[sorted_indices,:]
    if filenameI:
        cell_mean_spikeI=cell_mean_spikeI[sorted_indices,:]
        cell_sem_spikeI=cell_sem_spikeI[sorted_indices,:]
    all_spikes=all_spikes[1:,:] # get rid of the first row created by the np.empty()
    mean_all_spikes=np.mean(all_spikes,axis=0)
    sem_all_spikes=sem(all_spikes,axis=0)
    if filenameI:
        all_spikesI=all_spikesI[1:,:] # get rid of the first row created by the np.empty()
        mean_all_spikesI=np.mean(all_spikesI,axis=0)
        sem_all_spikesI=sem(all_spikesI,axis=0)
    # find and plot the average spikes for all cells
    plt.figure()
    plt.fill_between(np.arange(dt,tbeforespk+tafterspk+dt,dt),mean_all_spikes-sem_all_spikes,mean_all_spikes+sem_all_spikes,color='k',alpha=0.5)
    plt.plot(np.arange(dt,tbeforespk+tafterspk+dt,dt),mean_all_spikes,'k')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.axis([0,tbeforespk+tafterspk,-65,-45])
    if filenameI:
        plt.figure()
        plt.fill_between(np.arange(dt,tbeforespk+tafterspk+dt,dt),mean_all_spikesI-sem_all_spikesI,mean_all_spikesI+sem_all_spikesI,color='k',alpha=0.5)
        plt.plot(np.arange(dt,tbeforespk+tafterspk+dt,dt),mean_all_spikesI,'k')
        plt.xlabel(xlab)
        plt.ylabel('current [pA]')
#        plt.axis([0,tbeforespk+tafterspk,-65,-45])
    if filenameI:
        return mean_all_spikes, sem_all_spikes, mean_all_spikesI, sem_all_spikesI, cell_num_spikes, cell_mean_spike, cell_sem_spike, cell_mean_spikeI, cell_sem_spikeI, raster_times, v
    else:
        return mean_all_spikes, sem_all_spikes, cell_num_spikes, cell_mean_spike, cell_sem_spike, raster_times, v


# Plase-locking index of neurons (actual phase of cycle is arbitrary)
def PLI(raster_times,phasefreq=180,dt=0.05):
    num_cells=np.size(raster_times)
    wavelength=1/phasefreq # wavelength in s
    phase=[]
    unitvectors=[]
    all_phases=np.empty(1)
    all_unitvectors=np.empty((1,2))
    mean_phases=np.zeros(num_cells)
    mean_unitvectors=np.zeros((num_cells,2))
#    fig=plt.figure()
#    ax=fig.add_subplot(111,polar=True)
    plt.figure()
    for k in range(num_cells):
        num_spikes=np.size(raster_times[k])
        phase.append(np.zeros(num_spikes))
        unitvectors.append(np.zeros((num_spikes,2)))
        for p in range(num_spikes):
            phase[k][p]=((raster_times[k][p]*dt/1e3) % wavelength)*2*np.pi/wavelength # phase of the sin wave in s
            unitvectors[k][p,:]=np.array([np.cos(phase[k][p]),np.sin(phase[k][p])])
            plt.polar(np.array([0,phase[k][p]]),np.array([0,1]),'k')
        all_phases=np.concatenate((all_phases,phase[k]),axis=0)
        all_unitvectors=np.concatenate((all_unitvectors,unitvectors[k]),axis=0)
        mean_phases[k]=np.mean(phase[k],axis=0)
        mean_unitvectors[k]=np.mean(unitvectors[k],axis=0)
    all_phases=all_phases[1:] # get rid of the first row created by the np.empty()
    all_unitvectors=all_unitvectors[1:,:]
    mean_all_phase=np.mean(all_phases,axis=0)
    sem_all_phase=sem(all_phases,axis=0)
    mean_all_unitvector=np.mean(all_unitvectors,axis=0)
    sem_all_unitvector=sem(all_unitvectors,axis=0)
    mean_all_phaseb=np.arctan2(mean_all_unitvector[0],mean_all_unitvector[1]) # Get the mean phase from the mean unitvector
    plt.polar(np.array([0,phase[k][p]]),np.array([0,np.hypot(mean_all_unitvector[0],mean_all_unitvector[1])]),'darkorange',lw=2)
    return mean_all_unitvector, sem_all_unitvector, mean_all_phase, sem_all_phase, mean_all_phaseb, mean_phases, mean_unitvectors


## Plase-locking index of neurons using the Hilbert transform
#def PLI_hilbert(filenames,raster_times,dt=0.05,sim_time=10000):
#    num_cells=np.size(raster_times)
#    I=importvariable(filenames,num_cells=num_cells,sim_time=sim_time,dt=dt)
#    h_igaba=np.angle(hilbert(I[k,500/dt:])) #start 500 ms into the signal, and take the hilbert transform, outputting the phase in radians
#    #######START HERE#######
#    wavelength=1/phasefreq # wavelength in s
#    phase=[]
#    unitvectors=[]
#    all_phases=np.empty(1)
#    all_unitvectors=np.empty((1,2))
#    mean_phases=np.zeros(num_cells)
#    mean_unitvectors=np.zeros((num_cells,2))
##    fig=plt.figure()
##    ax=fig.add_subplot(111,polar=True)
#    plt.figure()
#    for k in range(num_cells):
#        num_spikes=np.size(raster_times[k])
#        phase.append(np.zeros(num_spikes))
#        unitvectors.append(np.zeros((num_spikes,2)))
#        for p in range(num_spikes):
#            phase[k][p]=((raster_times[k][p]*dt/1e3) % wavelength)*2*np.pi/wavelength # phase of the sin wave in s
#            unitvectors[k][p,:]=np.array([np.cos(phase[k][p]),np.sin(phase[k][p])])
#            plt.polar(np.array([0,phase[k][p]]),np.array([0,1]),'k')
#        all_phases=np.concatenate((all_phases,phase[k]),axis=0)
#        all_unitvectors=np.concatenate((all_unitvectors,unitvectors[k]),axis=0)
#        mean_phases[k]=np.mean(phase[k],axis=0)
#        mean_unitvectors[k]=np.mean(unitvectors[k],axis=0)
#    all_phases=all_phases[1:] # get rid of the first row created by the np.empty()
#    all_unitvectors=all_unitvectors[1:,:]
#    mean_all_phase=np.mean(all_phases,axis=0)
#    sem_all_phase=sem(all_phases,axis=0)
#    mean_all_unitvector=np.mean(all_unitvectors,axis=0)
#    sem_all_unitvector=sem(all_unitvectors,axis=0)
#    mean_all_phaseb=np.arctan2(mean_all_unitvector[0],mean_all_unitvector[1]) # Get the mean phase from the mean unitvector
#    plt.polar(np.array([0,phase[k][p]]),np.array([0,np.hypot(mean_all_unitvector[0],mean_all_unitvector[1])]),'darkorange',lw=2)
#    return mean_all_unitvector, sem_all_unitvector, mean_all_phase, sem_all_phase, mean_all_phaseb, mean_phases, mean_unitvectors


# Histogram of ISI times
def ISIhist(raster_times,dt=0.05,sim_time=10000,binnum=51,histmaxrange=10):
    ISI=[]
    ISI_all=np.empty((1))
    for k in range(np.size(raster_times)):
        ISI.append(np.diff(raster_times[k])*dt/1e3) #divide by 1e3 to get in seconds
        ISI_all=np.concatenate((ISI_all,ISI[k]),axis=0)
#        plt.figure()
#        n_cell, binedges_cell, _=plt.hist(ISI[k],bins=binnum,range=(0,histmaxrange),facecolor='k',edgecolor='w')
#        plt.xlabel('firing rate [spikes/s]')
#        plt.ylabel('number of neurons (out of 100)')
#        plt.axis([-2.5,histmaxrange+2.5,0,np.ceil(np.max(n_cell)/10)*10])
    ISI_all=ISI_all[1:]
    plt.figure()
    n, binedges, _=plt.hist(ISI_all,bins=binnum,range=(0,histmaxrange),facecolor='k',edgecolor='w')
    plt.yscale('log', nonposy='clip')
    plt.xlabel('firing rate [spikes/s]')
    plt.ylabel('number of neurons (out of 100)')
#    plt.axis([-2.5,histmaxrange+2.5,0,np.ceil(np.max(n)/10)*10])
    return ISI


# Spike triggered average for recovery time analysis
def STA_recoverytime(filenames,IPSP_times,num_cells=100,sim_time=10000,dt=0.05,reset=11,tbeforespk=5,tafterspk=25,xlab='time [ms]',ylab='membrane voltage [mV]'):
    #IPSP_times is a list the size of the number of cells, where each element of the list is an array of the IPSP times [in timesteps] in that cell
    sim_steps=sim_time/dt #timesteps
    v=importvariable(filenames,num_cells=num_cells,sim_time=sim_time,dt=dt)-70
    recoveryv=[]
    sta_IPSP_times=[]
    all_recoveryv=np.empty((1,(tbeforespk+tafterspk)/dt))
    recoveryt=np.empty((1,1))
    for o in range(num_cells):
        if np.size(IPSP_times[o])>1:
            sta_IPSP_times.append(IPSP_times[o][np.logical_and(IPSP_times[o]>tbeforespk/dt-1,IPSP_times[o]<sim_steps-tafterspk/dt)])
        elif np.size(IPSP_times[o])==0:
            sta_IPSP_times.append(np.array([]))
        elif IPSP_times[o]>tbeforespk/dt-1 and IPSP_times[o]<sim_steps-tafterspk/dt:
            sta_IPSP_times.append(IPSP_times[o])
        else:
            sta_IPSP_times.append(np.array([]))
        recoveryv_temp=np.zeros((np.size(sta_IPSP_times[o]),(tbeforespk+tafterspk)/dt))
        recoveryt_temp=np.zeros((np.size(sta_IPSP_times[o]),1))
        for k in range(np.size(sta_IPSP_times[o])):
            if np.size(sta_IPSP_times[o])==1:
                recoveryv_temp[k,:]=v[o,sta_IPSP_times[o]-tbeforespk/dt:sta_IPSP_times[o]+tafterspk/dt]
                minrecoveryv=np.min(recoveryv_temp[k,:])
                depthrecoveryv=v[o,sta_IPSP_times[o]]-minrecoveryv
            else:
                recoveryv_temp[k,:]=v[o,sta_IPSP_times[o][k]-tbeforespk/dt:sta_IPSP_times[o][k]+tafterspk/dt]
                minrecoveryv=np.min(recoveryv_temp[k,:])
                depthrecoveryv=v[o,sta_IPSP_times[o][k]]-minrecoveryv
            recoveryt_temp[k]=np.where(recoveryv_temp[k,:][tbeforespk/dt:]>=(minrecoveryv+0.632*depthrecoveryv))[0][0]#-tbeforespk/dt # <== adding this will give you the time including the before the spike section
            # find and plot the average spikes for each cell
            plt.figure()
            plt.plot(np.arange(dt,tbeforespk+tafterspk+dt,dt),recoveryv_temp[k,:],'k')
            plt.xlabel(xlab)
            plt.ylabel(ylab)
        recoveryv.append(recoveryv_temp)
        if o==0:
            recoveryt=recoveryt_temp
            all_recoveryv=recoveryv[o]
        else:
            recoveryt=np.vstack((recoveryt,recoveryt_temp))
            all_recoveryv=np.vstack((all_recoveryv,recoveryv[o]))
    recoveryfreq=1/(recoveryt*dt)
    mean_all_recoveryv=np.mean(all_recoveryv,axis=0)
    sem_all_recoveryv=sem(all_recoveryv,axis=0)
    mean_all_recoveryt=np.mean(recoveryt,axis=0)
    sem_all_recoveryt=sem(recoveryt,axis=0)
    mean_all_recoveryfreq=np.mean(recoveryfreq,axis=0)
    sem_all_recoveryfreq=sem(recoveryfreq,axis=0)
    # find and plot the average spikes for all cells
    plt.figure()
    plt.fill_between(np.arange(dt,tbeforespk+tafterspk+dt,dt),mean_all_recoveryv-sem_all_recoveryv,mean_all_recoveryv+sem_all_recoveryv,color='k',alpha=0.5)
    plt.plot(np.arange(dt,tbeforespk+tafterspk+dt,dt),mean_all_recoveryv,'k')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
#    plt.axis([0,tbeforespk+tafterspk,-65,-45])
    return mean_all_recoveryv, sem_all_recoveryv, mean_all_recoveryt, sem_all_recoveryt, mean_all_recoveryfreq, sem_all_recoveryfreq, recoveryv, all_recoveryv


# Spike triggered average for recovery time analysis using GABAergic events in baseline network
def STA_recoverytime_network(filename_voltage,filename_hreci,find_sem=False,num_cells=100,sim_time=10000,dt=0.05,tbeforespk=10,tafterspk=50,recoveryfraction=0.632,xlab='time [ms]',ylab='membrane voltage [mV]'):
    #IPSP_times is a list the size of the number of cells, where each element of the list is an array of the IPSP times [in timesteps] in that cell
    v=importvariable(filename_voltage,num_cells=num_cells,sim_time=sim_time,dt=dt)-70
    hreci=importvariable(filename_hreci,num_cells=num_cells,sim_time=sim_time,dt=dt)
    sta_hreci=hreci[:,tbeforespk/dt:-tafterspk/dt]
    total_sta_events=np.size(np.nonzero(sta_hreci)[0])
    mean_all_recoveryv=np.zeros((1,(tbeforespk+tafterspk)/dt))
    mean_recoveryv=np.zeros((num_cells,(tbeforespk+tafterspk)/dt))
    sem_recoveryv=np.zeros((num_cells,(tbeforespk+tafterspk)/dt))
    recoveryt=np.empty((1,1))
    for o in range(num_cells):
        ind_events_sta=np.nonzero(sta_hreci[o,:])[0]
        recoveryv_temp=np.zeros((np.size(ind_events_sta),(tbeforespk+tafterspk)/dt))
        recoveryt_temp=np.zeros((np.size(ind_events_sta),1))
        for i,k in enumerate(ind_events_sta+tbeforespk/dt):
            recoveryv_temp[i,:]=v[o,k-tbeforespk/dt:k+tafterspk/dt]/hreci[o,k] # divided by hreci[o,k] to correct for timesteps with multiple inputs
            recoveryt_temp[i]=recoverytimesteps(recoveryv_temp[i,:],dt=dt,tbeforespk=tbeforespk,tafterspk=tafterspk,recoveryfraction=recoveryfraction)
            mean_all_recoveryv+=recoveryv_temp[i,:]/total_sta_events
        mean_recoveryv[o,:]=np.mean(recoveryv_temp,axis=0)
        sem_recoveryv[o,:]=sem(recoveryv_temp,axis=0)
        if o==0:
            recoveryt=recoveryt_temp
        else:
            recoveryt=np.vstack((recoveryt,recoveryt_temp))
    if find_sem:
        ssq_all_recoveryv=np.zeros((1,(tbeforespk+tafterspk)/dt))
        for o in range(num_cells):
            ind_events_sta=np.nonzero(sta_hreci[o,:])[0]
            recoveryv_temp=np.zeros((1,(tbeforespk+tafterspk)/dt))
            for k in ind_events_sta+tbeforespk/dt:
                recoveryv_temp=v[o,k-tbeforespk/dt:k+tafterspk/dt]/hreci[o,k] # divided by hreci[o,k] to correct for timesteps with multiple inputs
                ssq_all_recoveryv+=(recoveryv_temp-mean_all_recoveryv)**2
        sem_all_recoveryv=np.sqrt(ssq_all_recoveryv/(total_sta_events-1))/np.sqrt(total_sta_events) # manually find the standard error
    else:
        sem_all_recoveryv=np.zeros((1,(tbeforespk+tafterspk)/dt))
    recoveryfreq=1/(recoveryt[recoveryt!=0]*dt/1e3)
    mean_all_recoveryt=np.mean(recoveryt[recoveryt!=0],axis=0)
    sem_all_recoveryt=sem(recoveryt[recoveryt!=0],axis=0)
    mean_all_recoveryfreq=np.mean(recoveryfreq,axis=0)
    sem_all_recoveryfreq=sem(recoveryfreq,axis=0)
    recoveryt_after=recoverytimesteps(mean_all_recoveryv,dt=dt,tbeforespk=tbeforespk,tafterspk=tafterspk,recoveryfraction=recoveryfraction)
    mean_recoveryfreq_after=1/(recoveryt_after*dt/1e3)
    # find and plot the average spikes for all cells
    plt.figure()
    plt.fill_between(np.arange(dt,tbeforespk+tafterspk+dt,dt),np.reshape(mean_all_recoveryv-sem_all_recoveryv,((tbeforespk+tafterspk)/dt,)),np.reshape(mean_all_recoveryv+sem_all_recoveryv,((tbeforespk+tafterspk)/dt,)),color='k',alpha=0.5)
    plt.plot(np.arange(dt,tbeforespk+tafterspk+dt,dt),np.reshape(mean_all_recoveryv,((tbeforespk+tafterspk)/dt,)),'k')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
#    plt.axis([0,tbeforespk+tafterspk,-65,-45])
    return mean_all_recoveryv, sem_all_recoveryv, mean_all_recoveryt, sem_all_recoveryt, mean_all_recoveryfreq, sem_all_recoveryfreq, mean_recoveryv, sem_recoveryv, mean_recoveryfreq_after


# Look at running average of average recovery voltage trace
def recoveryrunningavg(signal,windowsz=100,dt=0.05,tbeforespk=20,tafterspk=50,recoveryfraction=0.632,xlab='time [ms]',ylab='membrane voltage [mV]'):
    runningavg = rolling_mean(signal[0,:],windowsz)
    recoveryt = recoverytimesteps(runningavg,dt=dt,tbeforespk=tbeforespk,tafterspk=tafterspk,recoveryfraction=recoveryfraction) #,signal[0,:]
    recoveryfreq=1/(recoveryt[recoveryt!=0]*dt/1e3) # dt is in ms, so /1e3 to get it in seconds
    plt.figure()
    plt.plot(np.arange(dt,tbeforespk+tafterspk+dt,dt),runningavg,'k')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    return runningavg, recoveryt[recoveryt!=0], recoveryfreq


# Raster plot of spikes (uses the output of STA)
def ripplesraster(raster_times,num_cells=100,sim_time=10000,dt=0.05,titlelab='',xlab='time [ms]',ylab='neuron number'):
    plt.figure()
    for k in range(num_cells):
        plt.vlines(raster_times[k]*dt,k+0.5,k+1.5)
    plt.title(titlelab)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.axis([0,sim_time,0,num_cells+1])


# Sorted plot and histogram of neuron rates (uses the output of STA)
def ripplesrates(cell_num_spikes,sim_time=10000,binnum=51,histmaxrange=250):
    num_cells=np.size(cell_num_spikes)
    plt.figure()
    plt.plot(np.arange(num_cells)+1,cell_num_spikes/(sim_time/1e3),'k')
    plt.xlabel('neuron number (out of 100)')
    plt.ylabel('firing rate [spikes/s]')
    plt.axis([1,num_cells,0,np.ceil(np.max(cell_num_spikes/(sim_time/1e3))/10)*10])
    plt.figure()
    n, binedges, _=plt.hist(cell_num_spikes/(sim_time/1e3),bins=binnum,range=(0,histmaxrange),facecolor='k',edgecolor='w')
    plt.xlabel('firing rate [spikes/s]')
    plt.ylabel('number of neurons (out of 100)')
    plt.axis([-2.5,histmaxrange+2.5,0,np.ceil(np.max(n)/10)*10])
    return n, binedges


# Look at average firing rate before and after a step current
def steprate(rate,sim_time=1000,step_time=500,dt=1,xlab='time [ms]',ylab='network rate [spikes/neuron/s]',title='',color='k'):
    beforesteprate=np.mean(rate[step_time-100:step_time])
    aftersteprate=np.mean(rate[step_time:step_time+100])
#    plt.figure()
    plt.bar(np.arange(dt,sim_time+dt,dt),rate,color=color)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.axis([480,520,0,200])
    return beforesteprate, aftersteprate


# f-I curve of single neuron with current pulses
def fIcurve(filenames,sim_time=10000,dt=0.05,reset=11,tfirstpulse=1000,tlastpulse=9000,tpulse=500,first_current=5,last_current=45,num_steps=9):
    dpulse=(tlastpulse-tfirstpulse)/(num_steps-1)
    pulse_times=np.arange(tfirstpulse,tlastpulse+dpulse,dpulse)
    dI=(last_current-first_current)/(num_steps-1)
    currents=np.arange(first_current,last_current+dI,dI)
    v=np.loadtxt(filenames,delimiter=',',usecols=range(1))
    v-=70
    v[np.append(np.array(np.diff(np.array(np.logical_and(np.diff(v)==0,v[1:]==reset-70),dtype=int))>0,dtype=bool),np.array([0,0],dtype=bool))]=50 # set all time steps before refractory periods equal to 50 mV
    raster_times=np.where(v==50)[0] # find all spikes
    step_raster_times=[]
    ISI=[]
    frequencies=np.zeros(num_steps)
    for k,t in enumerate(pulse_times):
        step_raster_times.append(raster_times[np.logical_and(raster_times>=t/dt,raster_times<(t+tpulse)/dt)])
        if np.size(step_raster_times[k])>1:
            ISI.append(np.diff(step_raster_times[k])*dt/1e3) #divide by 1e3 to get in seconds
            frequencies[k]=np.mean(1/ISI[k])
        elif np.size(step_raster_times[k])==1:
            frequencies[k]=1/(tpulse/1e3) # divide by 1e3 to get in seconds (and thus Hz)
        else:
            ISI.append(np.array([]))
    coeffI = np.polyfit(currents,frequencies,1)
    gain = coeffI[0]
    fitline=np.poly1d(coeffI)
    # find and plot the average spikes for all cells
    plt.figure()
    plt.plot(currents,frequencies,'b.',label='raw data')
    plt.plot(currents,fitline(currents),'k',label='fitted line')
    plt.xlabel('current [pA]')
    plt.ylabel('spike rate [spikes/s]')
    plt.axis([currents[0],currents[-1],0,np.ceil(np.max(frequencies)/10)*10])
    return gain, currents, frequencies, fitline(currents)