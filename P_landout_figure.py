import pdb
import cPickle
import os

import copy

import numpy

import geometry.rotations

import thermal_field
import pilot_model
import sailplane
import state_record
import task
import state_machine

import collections

save_figure = True

import matplotlib
fontsize = 8
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=fontsize)
matplotlib.rc('ytick', labelsize=fontsize)
import matplotlib.pyplot as plt

mps = r'$(\mathrm{ms^{-1}})$'
kmh = r'$(\mathrm{kmh^{-1}})$'

n = numpy.arange(1, 11)
P_work = 0.5
P_landout = numpy.power(P_work, n)

f_landout = plt.figure(figsize=(4,2.5), dpi=300)
plt.plot(n, P_landout)
plt.grid()
plt.xlabel('number of options available')
plt.ylabel('P(landout)')
plt.tight_layout()

if save_figure:
    f_landout.savefig(
        'figures/p_landout.png', format='png')
else:
    plt.show()
