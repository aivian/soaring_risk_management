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

import collections

import matplotlib.pyplot as plt

#numpy.random.seed(int(numpy.pi * 100))

debug_plot = False

zi = 1000.0
wscale = 3.0
n_thermals = 1200

if 'field.p' not in os.listdir('.'):
    therm_field = thermal_field.ThermalField(
        100000.0, zi, 0.0, wscale, n_thermals, 0.5)
    with open('field.p', 'wb') as pfile:
        cPickle.dump(therm_field, pfile)
else:
    with open('field.p', 'rb') as pfile:
        therm_field = cPickle.load(pfile)
    #shutdown_idx = numpy.random.choice(
    #    numpy.arange(n_thermals), (int(0.3 * n_thermals),))
    #for idx in shutdown_idx:
    #    therm_field._thermals[idx]._w *= 0.0
#therm_field.plot()

state_history = state_record.StateRecord((10000, 5), (10000, 5))

turnpoints = numpy.array([
    [10.0, 10.0, 0.0],
    [50.0, 80.0, 0.0],
    [80.0, 10.0, 0.0],
    [10.0, 10.0, 0.0]]) * 1000.0
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

sailplane_pilot = pilot_model.SailplanePilot(polar, therm_field, task)

sailplane_pilot.set_mc(3.0)

i = 1
done = False
iter_max = 15e3

t = 0.0
last_state = ''
sh = []
max_risk = 0.0
while not done and i < iter_max and sailplane_sim.state[2] < 0:
    done, turnpoint_reached = task.update(sailplane_sim.state[:3])

    if done:
        continue

    risk_tolerance_met = sailplane_pilot.update_vehicle_state(
        sailplane_sim.state)
    state_changed = sailplane_pilot.state != last_state
    if not risk_tolerance_met or state_changed or turnpoint_reached:
        destination, risk_met, plan = sailplane_pilot.plan_destination(
            sailplane_pilot._visited_thermals)
        sailplane_pilot.set_plan(plan)
        sailplane_pilot.navigate_plan(sailplane_sim.state[:3])

    if sailplane_pilot.state != 'thermal':
        sailplane_pilot.navigate_plan(sailplane_sim.state[:3])
    else:
        sailplane_pilot.thermal()

    sh.append(sailplane_pilot.state)
    last_state = sailplane_pilot.state

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

    i += 1

    if i % 100 == 0:
        print('iter: {}, N: {:5.0f}, E: {:5.0f}, h: {:4.0f}, P: {:0.3f}'.format(
            i,
            sailplane_sim.state[0],
            sailplane_sim.state[1],
            -sailplane_sim.state[2],
            max_risk))
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

print('Task distance: {:3.1f} km'.format(task.distance() / 1000.0))
print('completed in {:3.1f} minutes'.format(t / 60.0))
print('speed: {:3.1f} m/s'.format(task.distance() / t))

cntr = collections.Counter(sh)
print(cntr)

plt.figure()
state_history.plot([2,])

plt.figure()
plt.plot(state_history.X[:,1], state_history.X[:,0], 'g')
for tp in turnpoints:
    plt.scatter(tp[1], tp[0], color='r', s=50)
plt.axis('equal')
therm_field.plot()
