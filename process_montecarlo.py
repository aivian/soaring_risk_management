import pdb
import cPickle
import os
import sys

import copy

import numpy
import scipy.io

import geometry.rotations

import thermal_field
import pilot_model
import sailplane
import state_record
import task as task_module
import state_machine as state_machine_module

import collections

save_fig = False

import matplotlib
fontsize = 8
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=fontsize)
matplotlib.rc('ytick', labelsize=fontsize)
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    assert os.path.isfile(sys.argv[1]), 'file {} not found'.format(sys.argv[1])
    with open(sys.argv[1], 'rb') as pfile:
        print('opening: {}'.format(sys.argv[1]))
        saves = cPickle.load(pfile)

    completed = 0
    failed = 0
    n_thermals = []
    speed = []
    z_range = []
    v_cruise = []
    for s in saves:
        thermals = []
        thermal_ids = []
        finite_states = numpy.array(s['finite_state_history'])
        if s['task'].finished:
            completed += 1
            distance = s['task'].distance()
            time = s['state_history'].t[-1]
            speed.append(distance / time)

            finite_states = numpy.array(s['finite_state_history'])
            thermal_idx = split_indices(finite_states == 'thermal')
            z_range.append(
                numpy.amax(s['state_history'].X[:,2]) -
                numpy.amin(s['state_history'].X[:,2]))
        else:
            failed += 1
            speed.append(0.0)

        optimize_idx = numpy.where(finite_states == 'optimize')[0]
        v_cruise.append(numpy.mean(
            s['state_history'].X[optimize_idx,4]))

        for thermal in s['thermal_field']._thermals:
            if thermal._w < 0.01:
                continue
            distances = numpy.linalg.norm(
                (thermal._x - s['state_history'].X[:,:3]) *
                numpy.array([1.0, 1.0, 0.0]), axis=1)
            if numpy.amin(distances) < 700:
                thermals.append(thermal)
                thermal_ids.append(thermal.id)
        n_thermals.append(len(thermals))

    polar_poly = numpy.array([
        -0.0028479077699783985,
        0.14645230406133683,
        -2.5118051525793175])
    speed = numpy.array(speed)
    n_thermals = numpy.array(n_thermals)
    z_range = numpy.array(z_range)
    v_cruise = numpy.array(v_cruise)
    LD = - v_cruise / numpy.polyval(polar_poly, v_cruise)
    density_1d = n_thermals / distance
    density_2d = density_1d * 2.0 / numpy.pi / 1000.0 / LD
    percolation_degree = density_2d * numpy.pi * numpy.power(1000.0 * LD, 2.0)

    if len(sys.argv) < 3:
        plot_run_idx = int(numpy.random.rand() * len(saves))
    else:
        plot_run_idx = sys.argv[2]
    if plot_run_idx < len(saves):

        plot_save = saves[plot_run_idx]
        plot_finite_states = numpy.array(plot_save['finite_state_history'])
        state_history = plot_save['state_history']

        thermal_idx = split_indices(plot_finite_states == 'thermal')
        minimize_risk_idx = split_indices(plot_finite_states == 'minimize_risk')
        optimize_idx = split_indices(plot_finite_states == 'optimize')
        final_glide_idx = split_indices(plot_finite_states == 'final_glide')

        f_barogram = plt.figure(figsize=(3,2.5), dpi=400)
        ax_baro = plt.axes()
        ax_baro.grid()

        f_map = plt.figure(figsize=(3,3), dpi=400)
        for idx in thermal_idx:
            plt.plot(
                state_history.X[idx,1] / 1000.0,
                state_history.X[idx,0] / 1000.0,
                'b', linewidth=2, label='thermal')
            ax_baro.plot(
                state_history.t[idx],
                -state_history.X[idx, 2],
                'b', linewidth=2, label='thermal')
        for idx in minimize_risk_idx:
            plt.plot(
                state_history.X[idx,1] / 1000.0,
                state_history.X[idx,0] / 1000.0,
                'c', linewidth=2, label='minimize risk')
            ax_baro.plot(
                state_history.t[idx],
                -state_history.X[idx, 2],
                'c', linewidth=2, label='minimize risk')
        for idx in optimize_idx:
            plt.plot(
                state_history.X[idx,1] / 1000.0,
                state_history.X[idx,0] / 1000.0,
                'g', linewidth=2, label='optimize')
            ax_baro.plot(
                state_history.t[idx],
                -state_history.X[idx, 2],
                'g', label='optimize')
        for idx in final_glide_idx:
            plt.plot(
                state_history.X[idx,1] / 1000.0,
                state_history.X[idx,0] / 1000.0,
                'r', linewidth=2, label='final glide')
            ax_baro.plot(
                state_history.t[idx],
                -state_history.X[idx, 2],
                'r', label='final glide')
        turnpoints = [tp.X for tp in plot_save['task']._turnpoints]
        for tp in turnpoints:
            plt.scatter(tp[1] / 1000.0, tp[0] / 1000.0, color='r', s=50)
        plt.axis('equal')
        plt.grid()
        plt.xlabel('east (m)', size=fontsize)
        plt.ylabel('north (m)', size=fontsize)
        ax_baro.set_xlabel('time (s)', size=fontsize)
        ax_baro.set_ylabel('altitude (m)', size=fontsize)
        f_map.tight_layout()
        f_barogram.tight_layout()
        plot_save['thermal_field'].plot(show=False)

    real_bins = numpy.linspace(
        numpy.amin(speed[speed>0]), numpy.amax(speed[speed>0]), 10)
    real_bins = numpy.linspace(20, 35, 15)
    bins = numpy.hstack((0.0, 1.0, real_bins))
    mat_data = {}
    mat_data['bins'] = bins
    mat_data['speeds'] = speed
    mat_data['density_1d'] = density_1d
    mat_data['density_2d'] = density_2d
    mat_data['percolation_degree'] = percolation_degree
    mat_file = os.path.splitext(sys.argv[1])[0] + '.mat'
    scipy.io.savemat(mat_file, mat_data)
    plt.figure()
    plt.hist(speed, bins, normed=True)

    finish_idx = numpy.where(speed > 0)[0]
    mean_speed = numpy.mean(speed[finish_idx])
    landout_percentage = 1.0 - float(len(finish_idx)) / float(len(speed))
    print('risk: {}, mean_speed: {}, landout_percentage: {}'.format(
        saves[0]['pilot']['P_landout_acceptable'],
        mean_speed,
        landout_percentage))

    if save_fig:
        P_work= s['thermal_field']._thermals[0].P_work
        risk = s['pilot']['P_landout_acceptable']
        shift = s['pilot']['gear_shifting']
        info_name = 'P_thermal={}-risk={}-shift={}'.format(
            P_work, risk, shift)
        info_name = info_name.replace('=', '_')
        info_name = info_name.replace('.', '_')
        f_barogram.savefig(
            'figures/sample_barogram-{}.png'.format(info_name),
            format='png')
        f_map.savefig(
            'figures/sample_flight_path-{}.png'.format(info_name),
            format='png')
        print('saving: {}'.format(info_name))
    #else:
        #plt.show()
