import numpy as np
import matplotlib as mt
mt.use('TKagg')
import matplotlib.pyplot as plt





def mufunc(q, zet):
    c= 3*10**(5.0) 
    H0 = 47.83 # in megaparsec aus (1.547+/-0.015)e-18 in si
    return 5.0*np.log10(  (c/H0)*zet + (c/ 2.0 * H0)*(1-q) * zet**2.0 ) +25


_,xd,yd,yer,_ = np.genfromtxt('hubbelrev.dat',unpack=True)
#xd,yd,yer = np.arange(1,40),np.arange(1,40), np.ones_like(np.arange(1,40))
x = np.linspace(np.min(xd),np.max(xd),20)
ybin = np.zeros((20,len(yd)))
xbin = np.zeros((20,len(xd)))
ybiner = np.zeros((20,len(xd)))
for i in range(1,len(x)):
    for j in range(len(xd)):
        if xd[j]>=x[i-1] and xd[j] < x[i]:
            ybin[i-1,j] = yd[j]
            xbin[i-1,j] = xd[j]
            ybiner[i-1,j] = yer[j]
        if i == (len(x)-1):
            if xd[j]>=x[i]:
                ybin[i,j] = yd[j]
                xbin[i,j] = xd[j]
                ybiner[i,j] = yer[j]

sigmasqr = ybiner*ybiner
sigmasqr = 1.0/sigmasqr
sigmasqr[np.isinf(sigmasqr)] = 0.0
ybin_mean = [np.dot(ybin[i,:],sigmasqr[i,:]) for i in range(len(x))]
sigmaar = np.sum(sigmasqr,axis=1)
ybin_mean = [ybin_mean[i]/sigmaar[i] for i in range(len(x))]
anzahl = np.zeros(len(x))
for i in range(len(x)):
    for j in xbin[i,:]:
        if j!=0:
            anzahl[i] = anzahl[i] + 1
#xbin_mean = [np.mean(xbin[i,:]) for i in range(len(x))]
xbin_mean = np.sum(xbin,axis=1)
xbin_mean = xbin_mean/anzahl
sigmaar = sigmaar/anzahl





qarray = np.linspace(-1, 10)

plt.figure(2)
summe= 0


for i in range(len(qarray)):
    summe = 0
    for j in range(len(xbin_mean)):
        summe += (1.0/len(xbin_mean)) * ( (mufunc(qarray[i], xbin_mean[j])-ybin_mean[j])/(sigmaar[j]) )**2.0
    print i
    plt.plot(qarray[i], summe, 'r+')






plt.figure(1)
print (x)
print (ybin_mean)
print ("xbin_mean = ", xbin_mean)
plt.plot(x,ybin_mean, 'r+')
plt.show()

plt.figure(7)

plt.grid(True) 

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.xlabel(r"$\log_{10}{(z)}$", fontsize=18)
plt.ylabel(r"$\mu$", fontsize=18)



plt.errorbar(np.log10(xbin_mean), ybin_mean, yerr=sigmaar, fmt='--o')


plt.plot(rlogval, model, label= '$\mu_{0}$')
plt.plot(np.log10(zint),muConcordance, 'r-', label= '$\mu_{Con.}$')
plt.plot(np.log10(zint),muref, 'b-', label= '$\mu_{ref.}$')
plt.plot(np.log10(zint),muCC, 'k-', label= '$\mu_{CC}$')

plt.legend(loc=0, fontsize=12)
