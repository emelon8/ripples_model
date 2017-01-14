# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 21:17:41 2016

@author: eric
"""

import numpy as np

def LIF_neuron(dt,v0,taum,gL,Vrest,Vr,Vt,arp,arp_counter,se,con_matrix,num_pyr,num_inh,num_chat,sampa,sgaba,snmda,gampa,ggaba,gnmda,gampae,vampa,vgaba,vnmda,DC):
    v = v0.copy() # copy timestep i to make timestep i+1
    # sum up inputs from all connected neurons of each type
    sa = np.apply_along_axis(lambda m: np.dot(m,sampa),0,con_matrix[:num_pyr,:])
    sg = np.apply_along_axis(lambda m: np.dot(m,sgaba),0,con_matrix[num_pyr:num_pyr+num_inh,:])
    sn = np.apply_along_axis(lambda m: np.dot(m,snmda),0,con_matrix[:num_pyr,:])
#    sc = np.apply_along_axis(lambda m: np.dot(m,schat),0,con_matrix[num_pyr+num_inh:num_pyr+num_inh+num_chat,:])
    I = gL*(v0-Vrest) + se*gampae*(v0-vampa) + sa*gampa*(v0-vampa) + sg*ggaba*(v0-vgaba) + sn*gnmda*(v0-vnmda) - DC
    R = 1 / (gL + se*gampae + sa*gampa + sg*ggaba + sn*gnmda) # [MOhms]
    v_inf = Vrest-(I*R) # find steady-state voltage [mV]

    v[(v0==Vr) & (arp_counter<arp/dt)] = Vr # if refractory period is not over, continue refractory period
    arp_counter[(v0==Vr) & (arp_counter<arp/dt)] += 1 # increase refractory period counter

    v[(v0!=Vr) | (arp_counter>=arp/dt)] = v0[(v0!=Vr) | (arp_counter>=arp/dt)]*(1-(dt/taum)) + v_inf[(v0!=Vr) | (arp_counter>=arp/dt)]*(dt/taum) # if voltage is not during refractory period, integrate next voltage step

    v[v0>Vt] = Vr # if voltage exceeds threshold, set next step to reset voltage
    return v, arp_counter