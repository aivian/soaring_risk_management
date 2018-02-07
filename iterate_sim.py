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

import matplotlib.pyplot as plt

debug_plot = False

zi = 1000.0
wscale = 3.0
n_thermals = 800
P_work = 0.7

turnpoints = numpy.array([
    [10.0, 20.0, 0.0],
    [50.0, 80.0, 0.0],
    [80.0, 20.0, 0.0],
    [10.0, 20.0, 0.0]]) * 1000.0
turnpoint_radii = numpy.array([3.0, 0.5, 0.5, 3.0]) * 1000.0
task = task.AssignedTask(turnpoints, turnpoint_radii)
total_distance = numpy.sum(
    numpy.linalg.norm(numpy.diff(turnpoints, axis=0), axis=1))

X0 = numpy.array([10000.0, 10000.0, -1000.0, 0.0, 30.0])
# N, E, D, psi, v

aircraft_parameters = {'Sref': 1.0, 'mass': 1.0}
sim_params = {'dt': 1.0}

polar_poly = numpy.array(
    [-0.0028479077699783985, 0.14645230406133683, -2.5118051525793175])
polar = sailplane.QuadraticPolar(polar_poly, 5.0, 100.0, 1.0, 1.0)

saves = []

for run_idx in range(1):
    therm_field = thermal_field.ThermalField(
        100000.0, zi, 0.0, wscale, n_thermals, 0.7)

    state_history = state_record.StateRecord((10000, 5), (10000, 5))
    sailplane_sim = sailplane.SailplaneSimulation(
        polar,
        aircraft_parameters,
        X0,
        state_history,
        sim_params)

    sailplane_pilot = pilot_model.SailplanePilot(polar, therm_field, task)
    sailplane_pilot.set_mc(3.0)

    state_machine = state_machine.create_optimize_machine(
        therm_field,
        sailplane_pilot,
        sailplane_pilot._navigator,
        polar,
        task)
    state_machine.set_state('prestart')

    i = 1
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
    while not done and i < iter_max and sailplane_sim.state[2] < 0:
        done, turnpoint_reached = task.update(sailplane_sim.state[:3])

        if done:
            continue

        sailplane_pilot.update_vehicle_state(
            sailplane_sim.state, state_machine.state)
        state, transition = state_machine.execute()

        if last_transition is not None:
            print('idx: {}, {}, {}, {}'.format(
                i,
                last_state,
                last_transition,
                sailplane_pilot._navigator._destination))
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

        i += 1

        if i % 100 == 0:
            print('iter: {}, N: {:5.0f}, E: {:5.0f}, h: {:4.0f}, P: {:0.3f}'.format(
                i,
                sailplane_sim.state[0],
                sailplane_sim.state[1],
                -sailplane_sim.state[2],
                max_risk) +
                ', state: {}'.format(state_machine.state))
            max_risk = 0.0

        risk = sailplane_pilot.check_plan()
        if risk > max_risk:
            max_risk = risk

        if i % 1000 == 0 and False:
            x = state_history.X
            plt.scatter(destination[1], destination[0])
            plt.plot(x[:,1], x[:,0])
            plt.quiver(
                x[-1,1], x[-1,0],
                acceleration_command[1], acceleration_command[0],
                color='r', linewidth=1)
            plt.quiver(
                x[-1,1], x[-1,0],
                i_V[1], i_V[0],
                color='b', linewidth=1)
            plt.axis('equal')
            plt.show()

    i -= 1

    save_data = {
        'thermal_field': therm_field,
        'state_history': state_history,
        'finite_state_history': sh,
        'task': task,
        }

    saves.append(save_data)

with open('save_data.p', 'wb') as pfile:
    cPickle.dump(saves, pfile)


distance = task.progress(sailplane_sim.state[:3])
print('Task distance: {:3.1f} km'.format(distance / 1000.0))
if task.finished:
    print('completed in {:3.1f} minutes'.format(t / 60.0))
else:
    print('landed out in {:3.1f} minutes'.format(t / 60.0))
print('speed: {:3.1f} m/s'.format(distance / t))

cntr = collections.Counter(sh)
print(cntr)

sh = numpy.array(sh)

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

thermal_idx = split_indices(sh == 'thermal')
minimize_risk_idx = split_indices(sh == 'minimize_risk')
optimize_idx = split_indices(sh == 'optimize')
final_glide_idx = split_indices(sh == 'final_glide')

plt.figure()
ax_baro = plt.axes()

plt.figure()
plt.plot(state_history.t, sc)
state_history.plot([4,])
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('speed (m/s)')
plt.legend(('command', 'flown'))

plt.figure()
for idx in thermal_idx:
    plt.plot(
        state_history.X[idx,1], state_history.X[idx,0], 'b', linewidth=2)
    ax_baro.plot(state_history.t[idx], -state_history.X[idx, 2], 'b')
for idx in minimize_risk_idx:
    plt.plot(state_history.X[idx,1], state_history.X[idx,0], 'c', linewidth=2)
    ax_baro.plot(state_history.t[idx], -state_history.X[idx, 2], 'c')
for idx in optimize_idx:
    plt.plot(state_history.X[idx,1], state_history.X[idx,0], 'g', linewidth=2)
    ax_baro.plot(state_history.t[idx], -state_history.X[idx, 2], 'g')
for idx in final_glide_idx:
    plt.plot(state_history.X[idx,1], state_history.X[idx,0], 'r', linewidth=2)
    ax_baro.plot(state_history.t[idx], -state_history.X[idx, 2], 'r')
for tp in turnpoints:
    plt.scatter(tp[1], tp[0], color='r', s=50)
plt.axis('equal')
plt.grid()
therm_field.plot(save=True)
