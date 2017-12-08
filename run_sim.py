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

import matplotlib.pyplot as plt

numpy.random.seed(int(numpy.pi * 100))

debug_plot = False

zi = 1600.0
wscale = 3.0
n_thermals = 2500

if 'field.p' not in os.listdir('.'):
    therm_field = thermal_field.ThermalField(
        100000.0, zi, 0.0, wscale, n_thermals, 1.0)
    with open('field.p', 'wb') as pfile:
        cPickle.dump(therm_field, pfile)
else:
    with open('field.p', 'rb') as pfile:
        therm_field = cPickle.load(pfile)
    shutdown_idx = numpy.random.choice(
        numpy.arange(n_thermals), (int(0.3 * n_thermals),))
    for idx in shutdown_idx:
        therm_field._thermals[idx]._w *= 0.0
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

X0 = numpy.array([0.0, 0.0, -1000.0, 0.0, 30.0])

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
navigator = pilot_model.Navigator()

MC = 2.0

i = 1
done = False
iter_max = 10e3

t = 0.0
last_state = ''
sh = []
while not done and i < iter_max and sailplane_sim.state[2] < 0:
    done = task.update(sailplane_sim.state[:3])
    sailplane_pilot.update_location(sailplane_sim.state[:3])
    if sailplane_pilot.state != last_state:
        destination, risk_met, plan = sailplane_pilot.plan_destination()
        sailplane_pilot.set_plan(plan)
        navigator.set_plan(plan)
    else:
        navigator.set_destination(
            navigator.destination_from_plan(sailplane_sim.state[:3]),
            sailplane_sim.state[:3])
    sh.append(last_state)
    last_state = sailplane_pilot.state

    i_V = numpy.array([
        numpy.cos(sailplane_sim.state[3]),
        numpy.sin(sailplane_sim.state[3]),
        0.0]) * sailplane_sim.state[4]
    X = sailplane_sim.state[:3]
    h = -sailplane_sim.state[2]
    w = therm_field.w(X, h)
    wind = numpy.array([0.0, 0.0, -w])


    state = sailplane_sim.state
    inertial_direction = i_V / state[4]
    acceleration_command = navigator.command(i_V, copy.deepcopy(state[:3]))
    acceleration_normal_to_velocity = numpy.cross(
        numpy.cross(inertial_direction, acceleration_command),
        inertial_direction)
    acceleration_body_frame = numpy.dot(
        geometry.rotations.zrot(state[3]), acceleration_normal_to_velocity)
    acceleration_body_frame += (
        geometry.helpers.unit_vector(2) * 9.806)
    phi_command = numpy.arctan2(
        acceleration_body_frame[1], acceleration_body_frame[2])
    speed_command = sailplane_pilot.select_speed()
    speed_rate_command = (speed_command - state[4]) / 5.0
    theta_command = numpy.clip(
        -numpy.arcsin(speed_rate_command / 9.806),
        -numpy.pi / 6.0,
        numpy.pi / 6.0)

    t += sim_params['dt']
    sailplane_sim.step(t, phi_command, theta_command, wind)

    i += 1

    if i % 100 == 0:
        print('iter: {}, N: {:5.0f}, E: {:5.0f}, h: {:4.0f}'.format(
            i,
            sailplane_sim.state[0],
            sailplane_sim.state[1],
            -sailplane_sim.state[2]))

i -= 1

