#!/usr/bin/env python
import pdb
import os
import cPickle

import matplotlib.pyplot as plt
import numpy

import scipy.io

save_fig = True

mps = r'$(\mathrm{ms^{-1}})$'
kmh = r'$(\mathrm{kmh^{-1}})$'

P_work=0.4
shift=True

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
        '0.1': 'save_data_n=350_503b7a66aff626ae6c26b11144acebad_gear_shift',
        }

if P_work == 0.4 and shift:
    runs = {
        '0.001': 'a1278a9c03889aa8decd5ddb15a4da32_n=350_Pwork=0.4_Paccept=0.001_shift=True',
        '0.01': '41f62f18815564130b655a4d80e6cbf4_n=350_Pwork=0.4_Paccept=0.01_shift=True',
        '0.02': '8e21569949a40acf8458acecafd40a49_n=350_Pwork=0.4_Paccept=0.02_shift=True',
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
PP = []
for P, run in runs.iteritems():
    print('opening: {}'.format(run))
    matpath = 'MC_runs/{}.mat'.format(run)
    ppath = 'MC_runs/{}.p'.format(run)
    if os.path.isfile(matpath):
        hist_data = scipy.io.loadmat(matpath)
        speeds.append(numpy.array(hist_data['speeds'][0]))
    else:
        with open(ppath, 'rb') as pfile:
            saves = cPickle.load(pfile)
        ispeeds = []
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
bins = numpy.linspace(numpy.amin(speeds[speeds>0]), numpy.max(speeds), 10)
bins = numpy.hstack((0.0, 1.0, bins))

hist_speeds = []
n = speeds.shape[0]
f_bar = plt.figure()
for idx, speed in enumerate(speeds):
    hist_speeds.append(numpy.histogram(speed, bins, normed=True)[0])
    #plt.plot(bins[1:], hist_speeds[-1])
    widths = numpy.diff(bins) / (n+1)
    start = bins[:-1] + widths * idx + widths / 2.0
    binmean = bins[:-1] + numpy.diff(bins) / 2.0
    plt.bar(start[2:] * 3.6, hist_speeds[-1][2:], widths[2:] * 3.6)

#plt.hist(speeds, bins, normed=True)
plt.grid()
plt.legend(['risk tolarance={}'.format(p) for p in PP])
#plt.legend()
plt.xlabel('speed {}'.format(kmh))
plt.ylabel('p(speed)')

f_line = plt.figure()
fmta = ['-b', '--g', '-.m', ':c', '-y']
for hs, P, fmt in zip(hist_speeds, PP, fmta):
    #plt.plot(bins, numpy.hstack((hs[0], 0.0, hs[1:])))
    plt.plot(binmean[2:] * 3.6, hs[2:], fmt, label='risk tolerance={}'.format(P))
plt.xlabel('speed {}'.format(kmh))
plt.ylabel('p(speed)')
plt.legend()
plt.grid()

f_landout = plt.figure()
hist_speeds = numpy.array(hist_speeds)
plt.plot(1.0 - PP, hist_speeds[:,0])
plt.xlabel('risk tolerance on each glide')
plt.ylabel('probability of landing out')
plt.grid()

if save_fig:
    info_name = 'P_thermal={}-shift={}'.format(P_work, shift)
    f_bar.savefig('figures/hist_bar-{}.png'.format(info_name), format='png')
    f_line.savefig('figures/hist_line-{}.png'.format(info_name), format='png')
    f_landout.savefig('figures/P_landout-{}.png'.format(info_name), format='png')
else:
    plt.show()
