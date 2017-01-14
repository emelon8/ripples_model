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

only_interneurons=True

if only_interneurons:
    gain_int,currents_int,freq_int,fitted_int=rp.fIcurve('voltage_int.txt')
else:
    gain_pyr,currents_pyr,freq_pyr,fitted_pyr=rp.fIcurve('voltage_pyr.txt')
    gain_int,currents_int,freq_int,fitted_int=rp.fIcurve('voltage_int.txt')