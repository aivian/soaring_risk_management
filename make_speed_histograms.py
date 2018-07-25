#!/usr/bin/env python
import pdb
import sys
import os
import pickle

save_fig = True
plot_fig = False

import matplotlib
fontsize = 8
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=fontsize)
matplotlib.rc('ytick', labelsize=fontsize)
import matplotlib.pyplot as plt
import numpy

import scipy.io

mps = r'$(\mathrm{m\ s^{-1}})$'
kmh = r'$(\mathrm{km\ h^{-1}})$'

P_work=0.7
shift=False

if len(sys.argv) == 3:
    P_work = float(sys.argv[1])
    shift = sys.argv[2] == 'True'

if P_work == 0.7 and not shift:
    P_work=0.7
    shift=False
    runs = {
        '0.001': 'save_data_n=350_83e199bfea718c143734d6f8150d0d55_',
        '0.01': 'save_data_n=350_be6f19475b2f6e3879c89f6473b57a29',
        '0.05': 'save_data_n=350_bce6028029501db08824082c408f3c2b',
        '0.1': 'save_data_n=350_16a6331660f8722ece6debf3bffc2d2c'
        }

if P_work == 0.7 and shift:
    runs = {
        '0.001': 'save_data_n=350_b4a43ec9f45019fc79b2ccdcb08a804e_gear_shift',
        '0.01': 'save_data_n=350_fee84f73d73ebd50e18ec00e1e100670_gear_shift',
        '0.05': 'save_data_n=350_6e53067f6235d5a1ccf9af2cca101b8b_gear_shift',
        '0.1': 'fa2e466d70918dd7810bfa1366de245f_n=350_Pwork=0.7_Paccept=0.1_shift=True'
        }

if P_work == 0.4 and shift:
    runs = {
        '0.001': 'a1278a9c03889aa8decd5ddb15a4da32_n=350_Pwork=0.4_Paccept=0.001_shift=True',
        '0.01': '41f62f18815564130b655a4d80e6cbf4_n=350_Pwork=0.4_Paccept=0.01_shift=True',
        #'0.02': '8e21569949a40acf8458acecafd40a49_n=350_Pwork=0.4_Paccept=0.02_shift=True',
        '0.05': '2caa537bd0bba07487c2a093786c588a_n=350_Pwork=0.4_Paccept=0.05_shift=True',
        '0.1': '5d306c28a4a6f87098053ec29477e534_n=350_Pwork=0.4_Paccept=0.1_shift=True'
        }

if P_work == 0.4 and not shift:
    runs = {
        '0.001': '50bb9194923d26af049a8e7aef897e01_n=350_Pwork=0.4_Paccept=0.001_shift=False',
        '0.01': '0eca4f1d9ed3d349db1b5a6175c2e6c1_n=350_Pwork=0.4_Paccept=0.01_shift=False',
        '0.05': '80c76ad2600887e3196e8db525e8acb6_n=350_Pwork=0.4_Paccept=0.05_shift=False',
        '0.1': 'ed92689c440d2f62fbb6eab271be41f6_n=350_Pwork=0.4_Paccept=0.1_shift=False'
        }

speeds = []
percolate_degree = []
density = []
PP = []
for P, run in runs.items():
    print('opening: {}'.format(run))
    matpath = 'MC_runs/{}.mat'.format(run)
    ppath = 'MC_runs/{}.p'.format(run)
    if os.path.isfile(matpath):
        hist_data = scipy.io.loadmat(matpath)
        speeds.append(numpy.array(hist_data['speeds'][0]))
        percolate_degree.append(hist_data['percolation_degree'])
        density.append(hist_data['density_1d'])
    else:
        with open(ppath, 'rb') as pfile:
            saves = pickle.load(pfile)
        ispeeds = []
        ipercolate = []
        for s in saves:
            if s['task'].finished:
                distance = s['task'].distance()
                time = s['state_history'].t[-1]
                ispeeds.append(distance / time)
            else:
                ispeeds.append(0.0)
        speeds.append(numpy.array(ispeeds))
        save_data = {'speeds': numpy.array(ispeeds)}
        scipy.io.savemat(matpath, save_data)
    PP.append(float(P))

PP = numpy.array(PP)
plot_order = numpy.argsort(PP)
PP = PP[plot_order]

speeds = numpy.array(speeds)[plot_order]
#bins = numpy.linspace(numpy.amin(speeds[speeds>0]), numpy.max(speeds), 10)
bins = numpy.linspace(60.0, 120.0, 15) * 10.0 / 36.0
bins = numpy.hstack((0.0, 1.0, bins))


thermal_spacing = []
for idx in plot_order:
    good_idx = numpy.where(
        numpy.logical_and(speeds[idx] > 0, density[idx] > 0))[0]
    thermal_spacing.append(1.0 / density[idx][0, good_idx] / 1000.0)

spacing_bins = numpy.linspace(
    numpy.amin(numpy.hstack(thermal_spacing)),
    numpy.max(numpy.hstack(thermal_spacing)),
    10)

hist_speeds = []
mean_speeds = []
n = speeds.shape[0]
f_bar = plt.figure(figsize=(4,3), dpi=400)
for idx, speed in enumerate(speeds):
    hist_speeds.append(numpy.histogram(speed, bins, density=True)[0])
    mean_speeds.append(numpy.mean(speed[speed > 0]))
    #plt.plot(bins[1:], hist_speeds[-1])
    widths = numpy.diff(bins) / (n+1)
    start = bins[:-1] + widths * idx + widths / 2.0
    binmean = bins[:-1] + numpy.diff(bins) / 2.0
    plt.bar(start[2:] * 3.6, hist_speeds[-1][2:], widths[2:] * 3.6)

hist_spacing = []
mean_spacing = []
for idx, spacing in enumerate(thermal_spacing):
    hist_spacing.append(
        numpy.histogram(spacing, spacing_bins, density=True)[0])
    mean_spacing.append(numpy.mean(spacing))
    spacing_widths = numpy.diff(spacing_bins) / (n + 1)
    spacing_start = (
        spacing_bins[:-1] + spacing_widths * idx + spacing_widths / 2.0)
    spacingbin = spacing_bins[:-1] + numpy.diff(spacing_bins) / 2.0

#plt.hist(speeds, bins, density=True)
plt.ylim([0, 0.4])
plt.grid()
plt.legend([r'$P_{tol}$' + '={}'.format(p) for p in PP])
#plt.legend()
plt.xlabel('speed {}'.format(kmh))
plt.ylabel('p(speed)')
plt.tight_layout()

f_line = plt.figure(figsize=(6,3), dpi=400)
fmta = ['-b', '--g', '-.m', ':c', '-y']
for hs, P, fmt in zip(hist_speeds, PP, fmta):
    #plt.plot(bins, numpy.hstack((hs[0], 0.0, hs[1:])))
    plt.plot(binmean[2:] * 3.6, hs[2:], fmt, label='risk tolerance={}'.format(P))
plt.xlabel('speed {}'.format(kmh))
plt.ylabel('p(speed)')
plt.legend()
plt.grid()
plt.tight_layout()

f_spacing = plt.figure(figsize=(4,3), dpi=400)
#for pp, P, fmt in zip(hist_percolate, PP, fmta):
#    plt.plot(percbin, pp, fmt, label='risk tolerance={}'.format(P))
#plt.plot([4.512, 4.512], [0.0, 0.3], '--k')
plt.semilogx(PP, mean_spacing,)
plt.semilogx([0.001, 0.1], [13.29, 13.29], 'k')
plt.xlabel(r'$P_{tol}$')
plt.ylabel(r'thermal spacing (km)')
#plt.ylabel('p(thermal frequency)')
#plt.legend()
plt.grid()
plt.tight_layout()

hist_speeds = numpy.array(hist_speeds)
mean_speeds = numpy.array(mean_speeds)

f_test = plt.figure()
plt.plot(mean_spacing, hist_speeds[:,0])
plt.xlabel(r'thermal spacing (km)')
plt.ylabel(r'$P_{landout}$')
plt.grid()

f_landout = plt.figure(figsize=(3,2), dpi=400)
plt.semilogx(PP, hist_speeds[:,0])
plt.xlabel(r'$P_{tol}$')
plt.ylabel(r'$P_{landout}$')
plt.xticks([0.001, 0.01, 0.1], ['0.001', '0.01', '0.1'])
plt.grid()
if shift:
    plt.yticks(numpy.arange(0.0, 0.06, 0.01))
    plt.ylim((0.0, 0.05))
plt.tight_layout()

save_data_2 = {
    'P_tolerance': PP,
    'hist_speeds': hist_speeds,
    'mean_spacing': mean_spacing,
    'spacing': spacing,
    'mean_speed': mean_speeds,
    }
scipy.io.savemat(
    'collected_hist_data-P_thermal={}-shift={}.mat'.format(P_work, shift),
    save_data_2)

if save_fig:
    info_name = 'P_thermal={}-shift={}'.format(P_work, shift)
    info_name = info_name.replace('=', '_')
    info_name = info_name.replace('.', '_')
    f_bar.savefig(
        'figures/hist_bar-{}.png'.format(info_name), format='png', dpi=300)
    f_line.savefig(
        'figures/hist_line-{}.png'.format(info_name), format='png', dpi=300)
    f_landout.savefig(
        'figures/P_landout-{}.png'.format(info_name), format='png', dpi=300)
    f_spacing.savefig(
        'figures/thermal_intensity-{}.png'.format(info_name),
        format='png', dpi=300)
if plot_fig:
    plt.show()
