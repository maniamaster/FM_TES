# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 18:43:50 2016

@author: Kiera
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:12:11 2016--epgrade

@author: Kieran
"""

from scipy import stats

import numpy as np
import scipy as sp
import matplotlib as mpl    
import matplotlib.pyplot as plt

from numpy import sqrt, pi, exp, linspace, loadtxt
from lmfit import  Model






label_size = 20
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size 
plt.figure(figsize=(16,10))

x = np.arange(0., 3, 0.02)
v1=np.log10(x**(-4))
v2=np.log10(x**(-3))
v3=np.log10(x**(0))
line1, =plt.plot(x,v1, 'k', label="RMR")
line2, =plt.plot(x,v2, 'm', label="NRM")
line3, =plt.plot(x,v3, 'b', label="CC")
first_legend = plt.legend(handles=[line1, line2, line3], loc=0,prop={'size':20})
plt.ylabel(r'$log_{10}\rho (a)$', fontsize=20)
plt.grid(True)
plt.xlabel('cosmic scalefactor $a$', fontsize=20)
"plt.plot(dist, result.init_fit, 'k--')"
"plt.plot(dist, result.best_fit, 'r-')"
plt.show()