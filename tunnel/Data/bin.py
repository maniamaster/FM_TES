# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 13:47:51 2016

@author: jan
"""

import numpy as np
import matplotlib as mt
mt.use('TKagg')
import matplotlib.pyplot as plt



def mufunc(q, zet):
    c= 3*10**(5.0) 
    H0 = 47.83 # in megaparsec aus (1.547+/-0.015)e-18 in si
    return 5.0*np.log10(  (c/H0)*zet + (c/ 2.0 * H0)*(1-q) * zet**2.0 ) +25


_,xd,yd,yer,_ = np.genfromtxt('hubbelrev.dat',unpack=True)
x = np.linspace(np.min(xd),np.max(xd),20)
ybin = np.zeros((20,len(yd)))
xbin = np.zeros((20,len(yd)))
for i in range(1,len(x)):
    for j in range(len(xd)):
        if xd[j]>=x[i-1] and xd[j] < x[i]:
            ybin[i-1, j]   = yd[j]
            xbin[i-1, j]   = xd[j]
            
sigmasqr = yer*yer
sigmasqr = 1.0/sigmasqr
sigmasqr[np.isnan(sigmasqr)] = 0.0
xbin_mean = np.mean(xbin)
ybin_mean = [np.dot(ybin[i,:],sigmasqr) for i in range(len(x))]
ybin_mean = ybin_mean/np.sum(sigmasqr)    
val = (mufunc(-1, xbin_mean))**(2.0)    
print x
print ybin_mean
plt.figure(1)
plt.plot(x,ybin_mean, 'r+')
plt.figure(2)
plt.plot(sigmasqr, 'b+')
plt.show()
    

