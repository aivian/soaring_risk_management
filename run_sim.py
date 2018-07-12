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

save_plot = True

if save_plot:
    fontsize=8
    figsize=(3,3)
else:
    fontsize=14
    fitsize=(5,5)

matplotlib.rc('xtick', labelsize=fontsize)
matplotlib.rc('ytick', labelsize=fontsize)

debug_plot = False

pilot_characteristics = {
    'n_minimum': 1,
    'thermal_center_sigma': 400.0,
    'detection_range': 700.0,
    'P_landout_acceptable': 0.01,
    'final_glide_margin': 0.1,
    'gear_shifting': False
    }
environment_characteristics = {
    'zi': 1000.0,
    'sigma_zi': 0.0,
    'wscale': 3.0,
    'n_thermals': 800,
    'P_work': 0.4,
    'area_scale': 100000.0,
    }

if 'field2.p' not in os.listdir('.'):
    therm_field = thermal_field.ThermalField(
        environment_characteristics['area_scale'],
        environment_characteristics['zi'],
        environment_characteristics['sigma_zi'],
        environment_characteristics['wscale'],
        environment_characteristics['n_thermals'],
        environment_characteristics['P_work'])
    with open('field2.p', 'wb') as pfile:
        cPickle.dump(therm_field, pfile)
    #therm_field.plot()
else:
    with open('field2.p', 'rb') as pfile:
        therm_field = cPickle.load(pfile)

state_history = state_record.StateRecord((10000, 5), (10000, 5))

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
    state_machine = state_machine.create_pilot_machine(
        therm_field,
        sailplane_pilot,
        sailplane_pilot._navigator,
        polar,
        task)
else:
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

    #if not risk_tolerance_met or state_changed or turnpoint_reached:
    #    destination, risk_met, plan = sailplane_pilot.plan_destination(
    #        sailplane_pilot._visited_thermals)
    #    sailplane_pilot.set_plan(plan)
    #    sailplane_pilot.navigate_plan(sailplane_sim.state[:3])

    #if sailplane_pilot.state != 'thermal':
    #    sailplane_pilot.navigate_plan(sailplane_sim.state[:3])
    #else:
    #    sailplane_pilot.thermal()

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
        #plt.scatter(destination[1], destination[0])
        #plt.plot(x[:,1], x[:,0])
        #plt.quiver(
        #    x[-1,1], x[-1,0],
        #    acceleration_command[1], acceleration_command[0],
        #    color='r', linewidth=1)
        #plt.quiver(
        #    x[-1,1], x[-1,0],
        #    i_V[1], i_V[0],
        #    color='b', linewidth=1)
        #plt.axis('equal')
        #plt.show()

i -= 1

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

save_data = {
    'thermal_field': therm_field,
    'state_history': state_history,
    'finite_state_history': sh,
    'task': task,
    }
save_name = 'run_sim_data_P={}_shift={}_Pwork={}.p'.format(
    pilot_characteristics['P_landout_acceptable'],
    pilot_characteristics['gear_shifting'],
    environment_characteristics['P_work'])
with open(save_name, 'wb') as pfile:
    cPickle.dump(save_data, pfile)

print(a)

plt.figure()
ax_baro = plt.axes()

plt.figure()
plt.plot(state_history.t, sc)
state_history.plot([4,])
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('speed (m/s)')
plt.legend(('command', 'flown'))

plt.figure(figsize=(3,3))
for idx in thermal_idx:
    plt.plot(
        state_history.X[idx,1] / 1000.0,
        state_history.X[idx,0] / 1000.0,
        'b', linewidth=2)
    ax_baro.plot(state_history.t[idx], -state_history.X[idx, 2], 'b')
for idx in minimize_risk_idx:
    plt.plot(
        state_history.X[idx,1] / 1000.0,
        state_history.X[idx,0] / 1000.0,
        'c', linewidth=2)
    ax_baro.plot(state_history.t[idx], -state_history.X[idx, 2], 'c')
for idx in optimize_idx:
    plt.plot(
        state_history.X[idx,1] / 1000.0,
        state_history.X[idx,0] / 1000.0,
        'g', linewidth=2)
    ax_baro.plot(state_history.t[idx], -state_history.X[idx, 2], 'g')
for idx in final_glide_idx:
    plt.plot(
        state_history.X[idx,1] / 1000.0,
        state_history.X[idx,0] / 1000.0,
        'r', linewidth=2)
    ax_baro.plot(state_history.t[idx], -state_history.X[idx, 2], 'r')
for tp in turnpoints:
    plt.scatter(
        tp[1] / 1000.0, tp[0] / 1000.0, color='r', s=30, edgecolors='none')
plt.axis('equal')
plt.xlabel('East (km)', fontsize=fontsize)
plt.ylabel('North (km)', fontsize=fontsize)
plt.tight_layout()
plt.grid()
therm_field.plot(save=save_plot)

