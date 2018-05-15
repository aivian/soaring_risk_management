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
import task as task_module
import state_machine as state_machine_module

import collections

import matplotlib.pyplot as plt

import hashlib

run_max = 350

debug_plot = False

turnpoints = numpy.array([
    [10.0, 20.0, 0.0],
    [50.0, 80.0, 0.0],
    [80.0, 20.0, 0.0],
    [10.0, 20.0, 0.0]]) * 1000.0
turnpoint_radii = numpy.array([3.0, 0.5, 0.5, 3.0]) * 1000.0
total_distance = numpy.sum(
    numpy.linalg.norm(numpy.diff(turnpoints, axis=0), axis=1))

# N, E, D, psi, v

aircraft_parameters = {'Sref': 1.0, 'mass': 1.0}
sim_params = {'dt': 1.0}

polar_poly = numpy.array(
    [-0.0028479077699783985, 0.14645230406133683, -2.5118051525793175])
polar = sailplane.QuadraticPolar(polar_poly, 5.0, 100.0, 1.0, 1.0)

saves = []

#use [0.005, 0.01, 0.05, 0.1, 0.2] for landout percentages
pilot_characteristics = {
    'n_minimum': 1,
    'thermal_center_sigma': 400.0,
    'detection_range': 700.0,
    'P_landout_acceptable': 0.001,
    'final_glide_margin': 0.1,
    'gear_shifting': False,
    }
environment_characteristics = {
    'zi': 1000.0,
    'sigma_zi': 0.0,
    'wscale': 3.0,
    'n_thermals': 800,
    'P_work': 0.7,
    'area_scale': 100000.0,
    }

for run_idx in range(run_max):
    task = task_module.AssignedTask(turnpoints, turnpoint_radii)

    therm_field = thermal_field.ThermalField(
        environment_characteristics['area_scale'],
        environment_characteristics['zi'],
        environment_characteristics['sigma_zi'],
        environment_characteristics['wscale'],
        environment_characteristics['n_thermals'],
        environment_characteristics['P_work'])

    X0 = numpy.array([10000.0, 10000.0, -1000.0, 0.0, 30.0])
    state_history = state_record.StateRecord((10000, 5), (10000, 5))
    sailplane_sim = sailplane.SailplaneSimulation(
        polar,
        aircraft_parameters,
        X0,
        state_history,
        sim_params)

    sailplane_pilot = pilot_model.SailplanePilot(
        polar, therm_field, task, pilot_characteristics)
    sailplane_pilot.set_mc(3.0)

    if pilot_characteristics['gear_shifting']:
        state_machine = state_machine_module.create_pilot_machine(
            therm_field,
            sailplane_pilot,
            sailplane_pilot._navigator,
            polar,
            task)
    else:
        state_machine = state_machine_module.create_optimize_machine(
            therm_field,
            sailplane_pilot,
            sailplane_pilot._navigator,
            polar,
            task)

    state_machine.set_state('prestart')

    iteration = 0
    done = False
    iter_max = 15e3

    t = 0.0
    last_state = ''
    last_transition = ''
    sh = []
    sc = [X0[4],]
    tc = [0.0,]
    max_risk = 0.0
    destination = None
    while not done and iteration < iter_max and sailplane_sim.state[2] < 0:
        done, turnpoint_reached = task.update(sailplane_sim.state[:3])

        if done:
            continue

        sailplane_pilot.update_vehicle_state(
            sailplane_sim.state, state_machine.state)
        state, transition = state_machine.execute()

        last_state = state
        last_transition = transition

        destination = sailplane_pilot._navigator._destination

        X = sailplane_sim.state[:3]
        h = -sailplane_sim.state[2]
        w = therm_field.w(X, h)
        wind = numpy.array([0.0, 0.0, -w])

        t += sim_params['dt']
        sailplane_sim.step(
            t,
            sailplane_pilot.phi_command(wind),
            sailplane_pilot.theta_command(wind),
            wind)

        sh.append(state)
        sc.append(sailplane_pilot.select_speed(wind[2]))
        tc.append(sailplane_pilot.theta_command(wind))

        iteration += 1

        if iteration % 1000 == 0:
            print('iter: {}, N: {:5.0f}, E: {:5.0f}, h: {:4.0f}, P: {:0.3f}'.format(
                iteration,
                sailplane_sim.state[0],
                sailplane_sim.state[1],
                -sailplane_sim.state[2],
                max_risk) +
                ', state: {}'.format(state_machine.state))
            max_risk = 0.0

        risk = sailplane_pilot.check_plan()
        if risk > max_risk:
            max_risk = risk

    save_data = {
        'thermal_field': therm_field,
        'state_history': state_history,
        'finite_state_history': sh,
        'task': task,
        'pilot': pilot_characteristics,
        'environment': pilot_characteristics,
        }
    print('finished run {} of {}'.format(run_idx+1, run_max))

    saves.append(save_data)

id = hashlib.md5(str(saves)).hexdigest()
if pilot_characteristics['gear_shifting']:
    shift = 'gear_shift'
else:
    shift=''
save_path = 'MC_runs/save_data_n={}_{}_{}.p'.format(run_max, id, shift)
with open(save_path, 'wb') as pfile:
    cPickle.dump(saves, pfile)

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

