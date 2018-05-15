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

import matplotlib
import matplotlib.pyplot as plt

#numpy.random.seed(int(numpy.pi * 100))

fontsize=8
figsize=(3,3)

matplotlib.rc('xtick', labelsize=fontsize)
matplotlib.rc('ytick', labelsize=fontsize)

with open('run_sim_data_3.p', 'rb') as pfile:
    risk_minimize = cPickle.load(pfile)
with open('run_sim_data_4.p', 'rb') as pfile:
    gear_shift = cPickle.load(pfile)

plt.figure(figsize=(3,3))
plt.plot(
    risk_minimize['state_history'].X[:,1] / 1000.0,
    risk_minimize['state_history'].X[:,0] / 1000.0,
    'g', linewidth=2, label='risk-minimizing')
plt.plot(
    gear_shift['state_history'].X[:,1] / 1000.0,
    gear_shift['state_history'].X[:,0] / 1000.0,
    'r', linewidth=2, label='gear-shifting')

turnpoints = [tp.X for tp in gear_shift['task']._turnpoints]
for tp in turnpoints:
    plt.scatter(
        tp[1] / 1000.0, tp[0] / 1000.0, color='r', s=30, edgecolors='none')

plt.axis('equal')
plt.xlabel('East (km)', fontsize=fontsize)
plt.ylabel('North (km)', fontsize=fontsize)
plt.tight_layout()
plt.grid()
plt.legend(prop={'size': fontsize})
gear_shift['thermal_field'].plot(save=True)
