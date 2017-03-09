import numpy as np
import scipy as sp
import scipy.optimize as opt
import scipy.integrate as intg
import matplotlib.pyplot as plt
import glob
import time
import os
import math

#Latex Fonts for matplotlib:
fig_width_pt = 426.7914  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt*0.5  # width in inches
fig_height =fig_width*golden_mean*0.5       # height in inches
fig_size = [fig_width,fig_height]
plt.rc('font', **{'family':'serif', 'serif':['Computer Modern Roman']})
params = {'backend': 'pdf',
                    'axes.labelsize': 10,
                    'font.size': 10,
                    'legend.fontsize': 8,
                    'xtick.labelsize': 8,
                    'ytick.labelsize': 8,
                    'figure.figsize': fig_size,
                    'text.usetex': True}

plt.rcParams.update(params)

def create_path(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print directory + " created."
    else:
        print directory + " already exists."

# loads all data from filepath, data[i] is the i'th file. data[i][j,k] is the
# j/k-th element of file i.
def fetchdata(filepath):
    data = np.array([np.loadtxt(filepath + 'TD2_T1_5.txt'),
                     np.loadtxt(filepath + 'TD2_T2_0.txt'),
                     np.loadtxt(filepath + 'TD2_T2_5.txt'),
                     np.loadtxt(filepath + 'TD2_T3_0.txt'),
                     np.loadtxt(filepath + 'TD2_T3_5.txt'),
                     np.loadtxt(filepath + 'TD2_T4_2.txt')])

    return data


#finds lowest size of list of arrays
def find_lowest_size(corrs):
    siz = corrs[0].size
    for x in corrs:
        newsiz = x.size
        if newsiz < siz:
            siz = newsiz
    return siz

#trims arrays to common lowest size
def trim_arrays(arrays):
    x = find_lowest_size(arrays)
    res = np.empty((0, x))
    for arr in arrays:
        crop = np.resize(arr, (1, x))
        res = np.append(res, crop, axis=0)
    return res

#finds minimum of array
def find_max(times, values, T):

    m = np.amin(values[T/dt:])
    pos = np.argmin(values[T/dt:])
    times = times[T/dt:]
    time = times[pos]

    return m, time, pos


def gauss(x, a, b, c, d):
    return a * np.exp(-b * (x + d)**2) + c



start = time.time()
################################################################################
filepath = "./FM.TES Daten/"
directory = "./results/"
create_path(directory)

SL = fetchdata(filepath)
der[i] = [np.gradient(SL[i][:-1,1],np.diff(SL[i][:,0])) for i in range(5)]

###FEHLER ????


################################################################################
end = time.time()

print "time taken = " + str(end - start)
