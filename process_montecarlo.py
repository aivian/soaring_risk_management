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

import matplotlib.pyplot as plt

if __name__ == '__main__':
    assert os.path.isfile(sys.argv[1]), 'file {} not found'.format(sys.argv[1])
    with open(sys.argv[1], 'rb') as pfile:
        saves = cPickle.load(pfile)

    completed = 0
    failed = 0
    speed = []
    for s in saves:
        if s['task'].finished:
            completed += 1
            distance = s['task'].distance()
            time = s['state_history'].t[-1]
            speed.append(distance / time)
        else:
            failed += 1
            speed.append(0.0)
    speed = numpy.array(speed)

    if int(sys.argv[2]) < len(saves):

        plot_save = saves[int(sys.argv[2])]
        plot_finite_states = numpy.array(plot_save['finite_state_history'])
        state_history = plot_save['state_history']

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

        thermal_idx = split_indices(plot_finite_states == 'thermal')
        minimize_risk_idx = split_indices(plot_finite_states == 'minimize_risk')
        optimize_idx = split_indices(plot_finite_states == 'optimize')
        final_glide_idx = split_indices(plot_finite_states == 'final_glide')

        plt.figure()
        ax_baro = plt.axes()

        f_map = plt.figure(figsize=(3,3))
        for idx in thermal_idx:
            plt.plot(
                state_history.X[idx,1] / 1000.0,
                state_history.X[idx,0] / 1000.0,
                'b', linewidth=2, label='thermal')
            ax_baro.plot(state_history.t[idx], -state_history.X[idx, 2], 'b')
        for idx in minimize_risk_idx:
            plt.plot(
                state_history.X[idx,1] / 1000.0,
                state_history.X[idx,0] / 1000.0,
                'c', linewidth=2, label='minimize_risk')
            ax_baro.plot(state_history.t[idx], -state_history.X[idx, 2], 'c')
        for idx in optimize_idx:
            plt.plot(
                state_history.X[idx,1] / 1000.0,
                state_history.X[idx,0] / 1000.0,
                'g', linewidth=2, label='optimize')
            ax_baro.plot(state_history.t[idx], -state_history.X[idx, 2], 'g')
        for idx in final_glide_idx:
            plt.plot(
                state_history.X[idx,1] / 1000.0,
                state_history.X[idx,0] / 1000.0,
                'r', linewidth=2, label='final glide')
            ax_baro.plot(state_history.t[idx], -state_history.X[idx, 2], 'r')
        turnpoints = [tp.X for tp in plot_save['task']._turnpoints]
        for tp in turnpoints:
            plt.scatter(tp[1] / 1000.0, tp[0] / 1000.0, color='r', s=50)
        plt.axis('equal')
        plt.grid()
        plt.xlabel('East (m)')
        plt.ylabel('North (m)')
        plot_save['thermal_field'].plot(show=False)

    real_bins = numpy.linspace(
        numpy.amin(speed[speed>0]), numpy.amax(speed[speed>0]), 10)
    real_bins = numpy.linspace(20, 35, 15)
    bins = numpy.hstack((0.0, 1.0, real_bins))
    mat_data = {}
    mat_data['bins'] = bins
    mat_data['speeds'] = speed
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

    plt.show()
