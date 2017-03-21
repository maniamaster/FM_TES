# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:12:11 2016--epgrade

@author: Kieran
"""

from scipy import stats

import uncertainties as uc
from uncertainties import ufloat,unumpy
import numpy as np
import scipy as sp
import matplotlib as mpl    
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties.unumpy import tanh


from numpy import sqrt, pi, exp, linspace, loadtxt
from lmfit import  Model

def uplot(xdata, ydata, c='k', marker='s', ms=4, mew=1, lw=1, label=None):
    x  = uc.unumpy.nominal_values(xdata)
    y  = uc.unumpy.nominal_values(ydata)
    xerr = uc.unumpy.std_devs(xdata)
    yerr = uc.unumpy.std_devs(ydata)
    return plt.errorbar(x,y, yerr, xerr, fmt=' ', ms=ms, mew=mew, lw=lw, \
                           mfc=c, mec=c, ecolor=c, marker=marker, label=label)[0]
   

#data = loadtxt('TD2_RT.txt')
#U= data[:, 0]
#I = data[:, 1]




UNL1, INL1 = loadtxt('TD2_RT.txt', unpack=True)
USL2, ISL2 = loadtxt('TD2_T1_5.txt', unpack=True)
USL3, ISL3 = loadtxt('TD2_T2_0.txt', unpack=True)
USL4, ISL4 = loadtxt('TD2_T2_5.txt', unpack=True)
USL5, ISL5 = loadtxt('TD2_T3_0.txt', unpack=True)
USL6, ISL6 = loadtxt('TD2_T3_5.txt', unpack=True)
USL7, ISL7 = loadtxt('TD2_T4_2.txt', unpack=True)



#####################################PLots


plt.close('all')


plt.figure(1)

plt.ticklabel_format(axis='y',style='sci',scilimits=(1,4))
plt.grid(True)
plt.xlabel(r"$U [\mathrm{V}]$")
plt.ylabel(r"$I [\mathrm{A}]$")











def func(x, a, b):
    return a * x + b;





popt, pcov = curve_fit(func, UNL1, INL1, absolute_sigma=True, sigma =0.0001)
plt.plot(UNL1, func(UNL1, *popt), 'b-', label='fit')




plt.plot(UNL1, INL1, 'r+')



plt.savefig('1.pdf')



plt.figure(2)
plt.ticklabel_format(axis='y',style='sci',scilimits=(1,4))
plt.grid(True)



plt.xlabel(r"$U [\mathrm{V}]$")
plt.ylabel(r"$I [\mathrm{A}]$")



plt.xlim(-0.004, 0.004)
plt.ylim(-0.0004, 0.0004)


plt.plot(UNL1, INL1, '+', label='$\\approx$ 25 $C^{\circ}$ ')

plt.plot(USL2, ISL2, '.', label='1.5 K')
plt.plot(USL3, ISL3, '.', label='2.0 K')
plt.plot(USL4, ISL4, '.', label='2.5 K')
plt.plot(USL5, ISL5, '.', label='3.0 K')
plt.plot(USL6, ISL6, '.', label='3.5 K')
plt.plot(USL7, ISL7, '.', label='4.2 K')


plt.legend(loc=4)

plt.savefig('2.pdf')





plt.figure(3)
plt.ticklabel_format(axis='y',style='sci',scilimits=(1,4))
plt.grid(True)







plt.xlabel(r"$U [\mathrm{V}]$")
plt.ylabel(r"$dI/dU   [\mathrm{A/V}]$")






plt.plot(USL2[0:-1], np.gradient(ISL2[0:-1], np.diff(USL2))
, '.', label='1.5 K')
plt.plot(USL3[0:-1], np.gradient(ISL3[0:-1], np.diff(USL3))
, '.', label='2.0 K')
plt.plot(USL4[0:-1], np.gradient(ISL4[0:-1], np.diff(USL4))
, '.', label='2.5 K')
plt.plot(USL5[0:-1], np.gradient(ISL5[0:-1], np.diff(USL5))
, '.', label='3.0 K')
plt.plot(USL6[0:-1], np.gradient(ISL6[0:-1], np.diff(USL6))
, '.', label='3.5 K')
plt.plot(USL7[0:-1], np.gradient(ISL7[0:-1], np.diff(USL7))
, '.', label='4.2 K')


plt.legend()

plt.savefig('3.pdf')









listmaxleft=[np.argmax(    np.gradient(ISL2[0:-1], np.diff(USL2) )[0:100]
 ) ,
 np.argmax(               np.gradient(ISL3[0:-1], np.diff(USL3) )[0:100]
 ),
np.argmax(   np.gradient(ISL4[0:-1], np.diff(USL4) )[0:100]
 ),
np.argmax(  np.gradient(ISL5[0:-1], np.diff(USL5) )[0:100]
 ),
np.argmax( np.gradient(ISL6[0:-1], np.diff(USL6) )[0:100]
 ),
np.argmax( np.gradient(ISL7[0:-1], np.diff(USL7))[0:100]
 )]



listmaxright=[np.argmax(    np.gradient(ISL2[0:-1], np.diff(USL2) )[101:200]

 ) ,
 np.argmax(               np.gradient(ISL3[0:-1], np.diff(USL3) )[101:200]

 ),
np.argmax(   np.gradient(ISL4[0:-1], np.diff(USL4) )[101:200]

 ),
np.argmax(  np.gradient(ISL5[0:-1], np.diff(USL5) )[101:200]

 ),
np.argmax( np.gradient(ISL6[0:-1], np.diff(USL6) )[101:200]

 ),
np.argmax( np.gradient(ISL7[0:-1], np.diff(USL7))[101:200]

 )]

Uleft= np.array(
[
USL2[0:-1][listmaxleft
[0]],
USL3[0:-1][listmaxleft
[1]],
USL4[0:-1][listmaxleft
[2]],
USL5[0:-1][listmaxleft
[3]],
USL6[0:-1][listmaxleft
[4]],
USL7[0:-1][listmaxleft
[5]]
]
)


Uright= np.array(
[
USL2[101:200][listmaxright
[0]],
USL3[101:200][listmaxright
[1]],
USL4[101:200][listmaxright
[2]],
USL5[101:200][listmaxright
[3]],
USL6[101:200][listmaxright
[4]],
USL7[101:200][listmaxright
[5]]
]
)

difference = np.abs(uc.unumpy.uarray(Uright,0.001
)-uc.unumpy.uarray(Uleft, 0.001) )

kb = 8.6173303*10**(-5.0)
TC = 7.2

value = difference/(kb*TC)


Tarr= uc.unumpy.uarray([1.5, 2.0, 2.5, 3.0, 3.5, 4.2], [0.075, 0.1, 0.125, 0.15, 0.175, 0.21])
Uf = uc.unumpy.uarray(Uright, 0.001)


def kor(U, T):
    elad = 1.0 #1.602*10**(-19)
    c1 = 1.113
    c2 = 2.107
    c3=2.138
    alpha = (elad*U - c1 * kb*T)**c3 - (c2*kb*T)**c3
    temp =  (alpha-(c2*kb*T)**c3 )**(1.0/c3)
    return temp







corrected = kor(Uf, Tarr)



def theo(T):
    c1 = 2.2345*10**(-22)
    TC= 7.193
    temp1= 1.74*uc.unumpy.sqrt( (TC/T) -1)
    temp2= tanh(temp1)
    return temp2

TC= 7.193





Trange= np.linspace(0.1, TC, 200)


plt.figure(6)
plt.xlabel(r"$T/Tc$")
plt.ylabel(r"$\Delta/\Delta0$")

plt.xlim(0,1)
plt.ylim(0,2.5)




plt.grid(True)
uplot(Tarr/TC, difference/(2.0*kb * 1.76* 7.193),c='b', label='Ohne Kor.')
uplot(Tarr/TC, corrected/(kb * 1.76* 7.193),c='r' , label='Mit Kor.')

plt.axhline(y=4.33/(1.76*2), color='m', linestyle='-')

plt.plot(Trange/TC, uc.unumpy.nominal_values(theo(Trange))+uc.unumpy.std_devs(theo(Trange)), 'g-')
plt.plot(Trange/TC, uc.unumpy.nominal_values(theo(Trange)), 'g-')
plt.plot(Trange/TC, uc.unumpy.nominal_values(theo(Trange))-uc.unumpy.std_devs(theo(Trange)), 'g-')

plt.axhline(y=1, color='black', linestyle='--')


plt.legend()

plt.savefig('4.pdf')


liter= ufloat(4.33, 0.1)
BCS = 1.76


difflit1= np.abs(liter-value)/(liter)


difflit2= np.abs(liter-(2*corrected)/(kb*TC) )/(liter)

comparison= np.abs( ( theo((Tarr))/(kb * 1.76* 7.193*1000)   - corrected/(kb * 1.76* 7.193))/ (theo(Tarr)/(kb * 1.76* 7.193*1000)))

plt.figure(7)
plt.xlabel(r"$T/Tc$")
plt.ylabel(r"Rel. Abw. BCS zu korr. Werten")


uplot(Tarr/TC, comparison)


plt.grid(True)


#for i in range(2,8):
#    j=str(i)
#    arr ='ISL'+ j
#    listmax.append(np.max(   np.gradient(ISL2[0:-1]               ))






#label_size = 20
#mpl.rcParams['xtick.labelsize'] = label_size
#mpl.rcParams['ytick.labelsize'] = label_size 
#plt.figure(figsize=(16,10))
#plt.grid(True)
#plt.plot(dist, vel+67.785*xco-236.245*yco+199.568*zco,         'ro', label='linear model')
#x = np.arange(0., 2.5, 0.02)
#v=465.2*x
#plt.plot(x,v,'b-', label='Hubbles data')
#plt.ylabel('Velocity [km/s]', fontsize=20)
#plt.xlabel('Distance [Mpc]', fontsize=20)
#plt.legend(loc=0, fontsize=20)
#"plt.plot(dist, result.init_fit, 'k--')"
#"plt.plot(dist, result.best_fit, 'r-')"
#plt.show()