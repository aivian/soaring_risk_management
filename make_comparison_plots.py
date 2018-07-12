#!/usr/bin/env python
import pdb
import os
import cPickle

save_fig = True

import matplotlib
fontsize = 8
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=fontsize)
matplotlib.rc('ytick', labelsize=fontsize)
import matplotlib.pyplot as plt
import numpy

import scipy.io

mps = r'$(\mathrm{ms^{-1}})$'
kmh = r'$(\mathrm{kmh^{-1}})$'

shift = [True, False]
P_work = [0.4, 0.7]

data = []
id_string = []
for P in P_work:
    for s in shift:
        data.append(scipy.io.loadmat(
            'collected_hist_data-P_thermal={}-shift={}.mat'.format(P, s)))
        if s:
            id_string.append(r'$P_{thermal}=$' + '{}'.format(P))
        else:
            id_string.append(None)


fmta = ['-b', '--b', '-r', '--r']

f_landout = plt.figure(figsize=(3,2), dpi=300)
for idx, d in enumerate(data):
    plt.semilogx(
        d['P_tolerance'][0], d['hist_speeds'][:,0],
        fmta[idx], label=id_string[idx])
plt.grid()
plt.xlabel(r'$P_{tolerance}$', size=fontsize)
plt.ylabel(r'$P_{landout}$', size=fontsize)
plt.legend(prop={'size': fontsize})
plt.tight_layout()

f_speed = plt.figure(figsize=(3,2), dpi=300)
for idx, d in enumerate(data):
    plt.semilogx(
        d['P_tolerance'][0], d['mean_speed'][0] * 3.6,
        fmta[idx], label=id_string[idx])
plt.grid()
plt.xlabel(r'$P_{tolerance}$', size=fontsize)
plt.ylabel('mean speed {}'.format(kmh), size=fontsize)
#plt.ylim([40, 110])
#plt.legend(prop={'size': fontsize})
plt.tight_layout()

if save_fig:
    f_landout.savefig('figures/landout_comparison.png', format='png')
    f_speed.savefig('figures/speed_comparison.png', format='png')
else:
    plt.show()
