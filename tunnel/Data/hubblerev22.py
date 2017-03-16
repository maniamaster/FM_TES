# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from scipy import stats

import numpy as np
import scipy as sp
import matplotlib as mpl    
import matplotlib.pyplot as plt
from pylab import *

import uncertainties as uc


from scipy.optimize import curve_fit,leastsq, minimize, fsolve
from scipy import signal, stats
from scipy.special import erf
from uncertainties import ufloat,unumpy
from uncertainties.unumpy import exp, log, sqrt








t1, t2, t3, t4, t5 = np.genfromtxt('hubbelrev.dat', unpack=True)

c= 3*10**(5)   # speed of light in mgparsec / second
z=t2
mu=t3
muerr=t4
H0 = 1.547*10**(-18)



logval= np.log10(t2)

def mufunc(q, zet):
    c= 3*10**(5.0) 
    H0 = 67.3 # in megaparsec aus (1.547+/-0.015)e-18 in si
    return 5.0*np.log10(  (c/H0)*zet + (c*(1-q)/ (2.0 * H0)) * zet**2.0 ) +25



slope, I, r_value, p_value, std_err = stats.linregress(logval,mu)



print ("r-squared:", r_value**2)


rlogval = np.linspace(-2, 1)

zint = np.linspace(0.001, 10, 50)

muConcordance  = mufunc(-0.52375, zint)
muref          = mufunc(0.5, zint)
muCC           = mufunc(-1.0, zint)


model = slope *rlogval+I

I = ufloat(I, std_err)

Hubble =1/( 10**(0.2*I-5.0) ) * ( 9.716*10**(-15.0) ) # H in 1/s
age = (1/Hubble) #year in sec



#here age in years
print  (Hubble, (age*3.171*10**(-8.0) ))

plt.figure(figsize=(16,10))
plt.figure(1)
plt.xlim(-2.0, 1)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

plt.xlabel(r"$\log_{10}{(z)}$", fontsize=35)
plt.ylabel(r"$\mu_{ obs. }$", fontsize=35)

plt.grid(True) 

plt.plot(rlogval, model,'r')
plt.plot(np.log10(zint),muConcordance, 'k-', label= '$\mu_{Con.}$')
plt.plot(np.log10(zint),muref, 'y-', label= '$\mu_{Ref.}$')
plt.plot(np.log10(zint),muCC, 'm-', label= '$\mu_{CC}$')

#CC passt nicht so schön die anderen beiden ungefähr gleich gut

plt.plot(logval, mu, '+b')
plt.legend(loc=0, fontsize=25)
plt.figure(figsize=(16,10))
plt.figure(2)
plt.xlim(-2.0, 1)
plt.grid(True) 
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel(r"$\log_{10}{(z)}$", fontsize=35)
plt.ylabel(r"$\Delta \mu= \vert \mu_i - \mu_{Ref} \vert $", fontsize=35)
diff1= np.abs(muConcordance-muref)
diff2= np.abs(muCC-muref)
plt.plot(np.log10(zint),diff1, 'b-',  label='$ \mu_i = \mu_{Con} $')
plt.plot(np.log10(zint),diff2, 'r-',  label='$ \mu_i = \mu_{CC} $')
plt.legend(loc=0, fontsize=25)





