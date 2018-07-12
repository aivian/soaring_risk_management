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

def split_indices(condition):
    """
    """
    base_idx = numpy.where(condition)[0]
    diff = numpy.diff(base_idx)
    splits = numpy.where(diff > 1)[0]
    if splits.shape == (0,):
        return [base_idx,]
    idx = []
    last_split = 0
    for split_idx in splits:
        idx.append(base_idx[last_split:split_idx])
        last_split = split_idx + 1
    idx.append(base_idx[splits[-1] + 1:])
    return idx

#with open('run_sim_data_P=0.001_shift=False.p', 'rb') as pfile:
#    risk_minimize = cPickle.load(pfile)
#with open('run_sim_data_P=0.001_shift=True.p', 'rb') as pfile:
#    gear_shift = cPickle.load(pfile)

with open('run_sim_data_P=0.01_shift=False_Pwork=0.4.p', 'rb') as pfile:
    risk_minimize = cPickle.load(pfile)
with open('run_sim_data_P=0.1_shift=False_Pwork=0.4.p', 'rb') as pfile:
    gear_shift = cPickle.load(pfile)

turnpoints = [tp.X for tp in gear_shift['task']._turnpoints]

sh = risk_minimize['finite_state_history']
thermal_idx = split_indices(sh == 'thermal')
minimize_risk_idx = split_indices(sh == 'minimize_risk')
optimize_idx = split_indices(sh == 'optimize')
final_glide_idx = split_indices(sh == 'final_glide')

f_flight_path_racing = plt.figure(figsize=(3,3), dpi=300)
for idx in thermal_idx:
    plt.plot(
        risk_minimize['state_history'].X[idx,1] / 1000.0,
        risk_minimize['state_history'].X[idx,0] / 1000.0,
        'b', linewidth=2)
for idx in optimize_idx:
    plt.plot(
        risk_minimize['state_history'].X[idx,1] / 1000.0,
        risk_minimize['state_history'].X[idx,0] / 1000.0,
        'g', linewidth=2)
for idx in final_glide_idx:
    plt.plot(
        risk_minimize['state_history'].X[idx,1] / 1000.0,
        risk_minimize['state_history'].X[idx,0] / 1000.0,
        'r', linewidth=2)
for tp in turnpoints:
    plt.scatter(
        tp[1] / 1000.0, tp[0] / 1000.0, color='r', s=30, edgecolors='none')
plt.axis('equal')
plt.xlabel('East (km)', fontsize=fontsize)
plt.ylabel('North (km)', fontsize=fontsize)
plt.tight_layout()
plt.grid()
gear_shift['thermal_field'].plot(show=False)

f_barogram_racing = plt.figure(figsize=(3, 2.5), dpi=300)
for idx in thermal_idx:
    plt.plot(
        risk_minimize['state_history'].t[idx],
        -risk_minimize['state_history'].X[idx,2],
        'b', linewidth=2)
for idx in optimize_idx:
    plt.plot(
        risk_minimize['state_history'].t[idx],
        -risk_minimize['state_history'].X[idx,2],
        'g', linewidth=2)
for idx in final_glide_idx:
    plt.plot(
        risk_minimize['state_history'].t[idx],
        -risk_minimize['state_history'].X[idx,2],
        'r', linewidth=2)
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('altitude (m)')
plt.tight_layout()

sh = gear_shift['finite_state_history']
if isinstance(sh, list):
    sh = numpy.array(sh)
thermal_idx = split_indices(sh == 'thermal')
minimize_risk_idx = split_indices(sh == 'minimize_risk')
optimize_idx = split_indices(sh == 'optimize')
final_glide_idx = split_indices(sh == 'final_glide')

f_flight_path_shifting = plt.figure(figsize=(3,3), dpi=300)
for idx in thermal_idx:
    plt.plot(
        gear_shift['state_history'].X[idx,1] / 1000.0,
        gear_shift['state_history'].X[idx,0] / 1000.0,
        'b', linewidth=2)
for idx in minimize_risk_idx:
    plt.plot(
        gear_shift['state_history'].X[idx,1] / 1000.0,
        gear_shift['state_history'].X[idx,0] / 1000.0,
        'c', linewidth=2)
for idx in optimize_idx:
    plt.plot(
        gear_shift['state_history'].X[idx,1] / 1000.0,
        gear_shift['state_history'].X[idx,0] / 1000.0,
        'g', linewidth=2)
for idx in final_glide_idx:
    plt.plot(
        gear_shift['state_history'].X[idx,1] / 1000.0,
        gear_shift['state_history'].X[idx,0] / 1000.0,
        'r', linewidth=2)
for tp in turnpoints:
    plt.scatter(
        tp[1] / 1000.0, tp[0] / 1000.0, color='r', s=30, edgecolors='none')
plt.axis('equal')
plt.xlabel('East (km)', fontsize=fontsize)
plt.ylabel('North (km)', fontsize=fontsize)
plt.tight_layout()
plt.grid()
gear_shift['thermal_field'].plot(show=False)

f_barogram_shifting = plt.figure(figsize=(3, 2.5), dpi=300)
for idx in thermal_idx:
    plt.plot(
        gear_shift['state_history'].t[idx],
        -gear_shift['state_history'].X[idx,2],
        'b', linewidth=2)
for idx in minimize_risk_idx:
    plt.plot(
        gear_shift['state_history'].t[idx],
        -gear_shift['state_history'].X[idx,2],
        'c', linewidth=2)
for idx in optimize_idx:
    plt.plot(
        gear_shift['state_history'].t[idx],
        -gear_shift['state_history'].X[idx,2],
        'g', linewidth=2)
for idx in final_glide_idx:
    plt.plot(
        gear_shift['state_history'].t[idx],
        -gear_shift['state_history'].X[idx,2],
        'r', linewidth=2)
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('altitude (m)')
plt.tight_layout()

if save_figure:
    f_flight_path_racing.savefig(
        'figures/sample_flight_path_racing.png', format='png')
    f_barogram_racing.savefig(
        'figures/sample_barogram_racing.png', format='png')
    f_flight_path_shifting.savefig(
        'figures/sample_flight_path_shifting.png', format='png')
    f_barogram_shifting.savefig(
        'figures/sample_barogram_shifting.png', format='png')
else:
    plt.show()
