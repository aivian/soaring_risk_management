import pdb
import cPickle
import os

import copy

import numpy

import thermal_field

import matplotlib.pyplot as plt

numpy.random.seed(int(numpy.pi * 100))

debug_plot = False

zi = 1000.0
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
therm_field.plot()

wp = numpy.array([
    [10.0, 10.0],
    [50.0, 80.0],
    [80.0, 10.0],
    [10.0, 10.0]]) * 1000.0
wp_distance = numpy.linalg.norm(numpy.diff(wp, axis=0), axis=1)
total_distance = numpy.sum(wp_distance)
next_wp = wp[0]
wp_idx = 0

X0 = numpy.array([10.0, 10.0])
h0 = 1000.0
X = X0
h = h0

polar = numpy.array(
    [-0.0028479077699783985, 0.14645230406133683, -2.5118051525793175])
a = polar[0]
b = polar[1]
c = polar[2]

dt = 3.0
Xhist = numpy.zeros((50000,2))
Vhist = numpy.zeros((50000,))
hhist = numpy.zeros((50000,))
whist = numpy.zeros((50000,))

done = False
iter_max = 10e3

i = 1
MC = 2.0

min_connection = 1

detect_range = 10000.0

start_time = 0.0

thermalling = False
needs_destination = True
seeking_thermal = False
look_for_thermal_interval = 100
new_destination = False

visited = []

while not done and i < iter_max and h > 0:
    w = therm_field.w(X, h)

    if polar[2] + w > 0:
        v = -polar[1] / 2.0 / polar[0]
    else:
        v = numpy.sqrt((polar[2] + w - MC) / polar[0])

    if (
        (needs_destination or
        i % look_for_thermal_interval == 0) and
        not seeking_thermal and
        not thermalling and
        not new_destination):
        therm_info = therm_field.get_thermals_in_range(
            X, detect_range, exclude=visited)
        rvectors, therms, therms_idx  = therm_info

        if len(therms) == 0:
            destination = next_wp
            needs_destination = False
            seeking_thermal = False
            new_destination = True

            if debug_plot:
                plt.plot(Xhist[:i,0], Xhist[:i,1])
                plt.scatter(wp[:,0], wp[:,1])
                plt.scatter(destination[0], destination[1], c='r')
                plt.xlabel('x position (m)')
                plt.xlabel('y position (m)')
                plt.title(i)
                plt.grid()
                plt.show()

            continue

        course_line = next_wp - X
        course_line /= numpy.linalg.norm(course_line)
        therm_ranges = numpy.linalg.norm(rvectors, axis=1)
        rvectors /= numpy.expand_dims(therm_ranges, 1)
        course_projection = numpy.dot(rvectors, course_line)
        if numpy.amax(course_projection) < 1.0 / numpy.sqrt(2.0):
            idx = numpy.argmax(course_projection)
            therms = [therms[idx],]
            rvectors = rvectors[idx:idx+1]
            therm_ranges = therm_ranges[idx:idx+1]
        else:
            sort_idx = numpy.argsort(-course_projection)
            away_idx = numpy.argmax(numpy.where(
                course_projection[sort_idx] > 1.0 / numpy.sqrt(2.0)))
            therms = [therms[idx] for idx in sort_idx[:away_idx+1]]
            therms_idx = [
                therms_idx[idx] for idx in sort_idx[:away_idx+1]]
            rvectors = rvectors[sort_idx[:away_idx+1]]
            therm_ranges = therm_ranges[sort_idx[:away_idx+1]]

        glide_range = -v / numpy.polyval(polar, v) * h
        closest = therms[numpy.argmin(therm_ranges)]
        closest_idx = therms_idx[numpy.argmin(therm_ranges)]
        if numpy.sum(therm_ranges < glide_range) < min_connection:
            next_thermal = closest
            next_thermal_idx = closest_idx
            destination = next_thermal.X
            needs_destination = False
            seeking_thermal = True
            new_destination = True

            if debug_plot:
                plt.plot(Xhist[:i,0], Xhist[:i,1])
                plt.scatter(wp[:,0], wp[:,1])
                plt.scatter(destination[0], destination[1], c='r')
                plt.xlabel('x position (m)')
                plt.xlabel('y position (m)')
                plt.title(i)
                plt.grid()
                plt.show()

            if (
                next_thermal.distance(X) >
                numpy.linalg.norm(X - next_wp)):
                destination = next_wp
                needs_destination = False
                seeking_thermal = False
                new_destination = True

            continue

        next_thermal = therms[0]
        next_thermal_idx = therms_idx[0]
        destination = therms[0].X
        needs_destination = False
        seeking_thermal = True
        new_destination = True

        if debug_plot:
            plt.plot(Xhist[:i,0], Xhist[:i,1])
            plt.scatter(wp[:,0], wp[:,1])
            plt.scatter(destination[0], destination[1], c='r')
            plt.xlabel('x position (m)')
            plt.xlabel('y position (m)')
            plt.title(i)
            plt.grid()
            plt.show()

        if next_thermal.distance(X) > numpy.linalg.norm(X - next_wp):
            destination = next_wp
            needs_destination = False
            seeking_thermal = False
            new_destination = True

        continue

    new_destination = False

    R = numpy.linalg.norm(X - destination)
    R = numpy.clip(R, 1.0, numpy.inf)

    xdot = -v * (X[0] - destination[0]) / R
    ydot = -v * (X[1] - destination[1]) / R
    hdot = numpy.polyval(polar, v) + w

    if R < 60.0 and seeking_thermal:
        thermalling = True
        needs_destination = False

    if (hdot > MC or thermalling):
        v = -b / 2.0 / a
        #rvector = next_thermal.vector_to(X)
        #rvector /= numpy.linalg.norm(rvector)
        #omega = v / 60.0
        #phi = numpy.arctan2(rvector[0], rvector[1])
        #next_phi = phi + omega * dt
        next_x = next_thermal.X + numpy.array([1.0, 0.0]) * 60.0
        xdot = (next_x[0] - X[0]) / dt
        ydot = (next_x[1] - X[1]) / dt
        hdot = 1.1 * numpy.polyval(polar, v) + w

    if (h > 0.9 * zi or hdot < 0.7 * MC) and thermalling:
        visited.append(next_thermal_idx)
        needs_destination = True
        seeking_thermal = False
        thermalling = False
        continue

    Xdot = numpy.array((xdot, ydot))

    X += dt * Xdot
    h += dt * hdot
    Xhist[i] = copy.deepcopy(X)
    Vhist[i] = copy.deepcopy(v)
    hhist[i] = copy.deepcopy(h)
    whist[i] = copy.deepcopy(w)

    if numpy.linalg.norm(next_wp - X) < 1.0e3:
        if wp_idx == (len(wp) - 1):
            done = True
            end_time = i * dt
        else:
            if wp_idx == 0:
                start_time = i * dt
            wp_idx += 1
            next_wp = wp[wp_idx]
            visited = []

    i += 1

    if i % 1000 == 0:
        course_distance = 0.0
        if wp_idx > 0:
            segment_distance = numpy.linalg.norm(wp[wp_idx] - X)
            course_distance += segment_distance
            course_distance += numpy.sum(wp_distance[:(wp_idx-1)])
        print('iteration: {}, progress: {}, h: {}'.format(
            i, course_distance / total_distance, h))

i -= 1

plt.plot(Xhist[:i,0], Xhist[:i,1])
plt.scatter(wp[:,0], wp[:,1])
plt.xlabel('x position (m)')
plt.xlabel('y position (m)')
plt.grid()
plt.axis('equal')
plt.show()

plt.plot(numpy.arange(i)*dt, hhist[:i])
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('altitude (m)')
plt.show()

plt.plot(numpy.arange(i)*dt, whist[:i])
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('updraft (m/s)')
plt.show()

print('speed: {}'.format(total_distance / (end_time - start_time)))
