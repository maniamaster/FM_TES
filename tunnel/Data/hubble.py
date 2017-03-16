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

def degtorad(angle):
    temp = 2*np.pi *((angle)/360)
    return temp

data = loadtxt('data.dat')
dist = data[:, 0]
vel = data[:, 1]
xco = np.cos(degtorad(data[:, 2]))*np.cos(degtorad(data[:, 3]))
yco = np.sin(degtorad(data[:, 2]))*np.cos(degtorad(data[:, 3]))
zco = np.sin(degtorad(data[:, 3]))


    
def velcorr(dist, K, X, Y, Z):
    "1-d gaussian: gaussian(x, amp, cen, wid)"
    return (K*dist + X*xco + Y*yco + Z*zco)
gmod = Model(velcorr)
print("test")
result = gmod.fit(vel, dist=dist,xco=xco, yco=yco, zco=zco, K=500, X=0, Y=0, Z=0)

print(result.fit_report())

label_size = 20
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size 
plt.figure(figsize=(16,10))
plt.grid(True)
plt.plot(dist, vel+67.785*xco-236.245*yco+199.568*zco,         'ro', label='linear model')
x = np.arange(0., 2.5, 0.02)
v=465.2*x
plt.plot(x,v,'b-', label='Hubbles data')
plt.ylabel('Velocity [km/s]', fontsize=20)
plt.xlabel('Distance [Mpc]', fontsize=20)
plt.legend(loc=0, fontsize=20)
"plt.plot(dist, result.init_fit, 'k--')"
"plt.plot(dist, result.best_fit, 'r-')"
plt.show()