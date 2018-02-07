import pdb

import numpy
import shapely.geometry

import geometry.lines

import hashlib
import datetime

import matplotlib.pyplot as plt

class Thermal(object):
    """
    """
    def __init__(self, x, r, w, zi):
        """Constructor

        Arguments:
            x: location (m, n/e)
            r: radius (m)
            w: updraft strength (m/s)
            zi: top (m)

        Returns:
            class instance
        """
        self._x = x
        self._r = r
        self._w = w
        self._zi = zi

        self._id = hashlib.md5(
            str((x, r, w, zi, datetime.datetime.now()))).hexdigest()

    def w(self, xtest, ztest):
        """Get the updraft value at a location

        Arguments:
            xtest: location to test (m, n/e)
            ztest: altitue (m)

        Returns:
            w: updraft velocity (m/s, +up)
        """
        if ztest > self._zi:
            return 0.0

        R = numpy.linalg.norm(
            (self._x - xtest) * numpy.array([1.0, 1.0, 0.0]))
        w = numpy.exp(-R / self._r) * self._w
        z_scale = numpy.clip(ztest / self._zi - 1.0, 0.0, numpy.inf)
        w *= numpy.exp(-z_scale * 4.0)
        return w

    def distance(self, location):
        """Get distance from a point to the thermal

        Arguments:
            location: position to check

        Returns:
            distance: range to the thermal
        """
        return numpy.linalg.norm(self.vector_to(location))

    def vector_to(self, location):
        """Get a vector from a point to the thermal

        Arguments:
            location: position to check

        Returns:
            vector: vector from that point to the thermal
        """
        return (self.X - location) * numpy.array([1.0, 1.0, 0.0])

    @property
    def X(self):
        """Get the location
        """
        return self._x

    @property
    def point(self):
        """Get the location as a shapely point
        """
        return shapely.geometry.Point(self._x[0], self._x[1])

    @property
    def id(self):
        """Get a unique identifier for the thermal
        """
        return self._id

    @property
    def zmax(self):
        """Get the thermal max height
        """
        return self._zi

    def estimate_center(self, center_accuracy):
        """Guess the center point of a thermal from a distance

        Arguments:
            center_accuracy: the accuracy with which we can estimate where the
                center is

        Returns:
            estimated_center: where our best guess of the thermal center is
        """
        estimated_center = numpy.random.rand(3) * center_accuracy + self._x
        estimated_center[2] = 0.0
        return estimated_center

    def working(self, location):
        """Check if a thermal is working

        Arguments:
            location: where the aircraft is

        Returns:
            working: True if the thermal works, False if it doesn't, None if we
                aren't close enough to tell
        """
        # say we can detect a thermal if the updraft is 5% of it's peak...
        R = numpy.linalg.norm(
            (self._x - location) * numpy.array([1.0, 1.0, 0.0]))
        min_distance = -numpy.log(0.05) * self._r
        if R < min_distance:
            return self._w > 0
        else:
            return None

class ConvergenceLine(object):
    """
    """
    def __init__(self, x, r, w, zi):
        """Constructor

        Arguments:
            x: two element array of start and end points for the line (m, n/e)
            r: radius (m)
            w: updraft strength (m/s, +up)
            zi: top (m)

        Returns:
            class instance
        """
        self._x = x
        self._r = r
        self._w = w
        self._zi = zi

    def w(self, xtest):
        """Get the updraft value at a location

        Arguments:
            xtest: location to test (m, n/e)

        Returns:
            w: updraft velocity (m/s, +up)
        """
        xtest = numpy.array(xtest, ndmin=2)

        R = numpy.linalg.norm(
            geometry.lines.point_line_distance(xtest, line)[0])

        w = numpy.exp(R / self._r) * self._w
        w *= (numpy.cos(numpy.pi * ztest / self._zi) / 2.0 + 1.0)
        return w

class StochasticThermal(Thermal):
    """A class for a stochastic thermal.

    This is similar to the thermal except that it has some probability of
    not working. This probability can be specified and it's work / not work
    property can be evaluated either at initialization or later
    """
    def __init__(self, x, r, w, zi, P, realize=True):
        """Constructor

        Arguments:
            x: location (m, n/e)
            r: radius (m)
            w: updraft strength (m/s)
            zi: top (m)
            P: probability that this thermal works
            realize: optional bool. Realize this thermal (figure out if it works
                right away, or wait till later)

        Returns:
            class instance
        """
        super(StochasticThermal, self).__init__(x, r, w, zi)
        self._P = P
        if realize:
            self.realize()

    def realize(self):
        """Realize this thermal

        determine if it works or not

        Arguments:
            no arguments

        Returns:
            no returns
        """
        if numpy.random.rand() > self._P:
            self._w *= 0.0

    @property
    def P_work(self):
        """Get the probability that this thermal works
        """
        return self._P

class ThermalField(object):
    def __init__(self, x, zi, sigma_zi, wscale, n, P_work=None):
        """Constructor

        Arguments:
            x: limit of area to use, will be square x on a side
            zi: bl depth
            sigma_zi: if randomness in thermal depth is desired
            wscale: average thermals strength
            n: number of thermals to populate
            P: probability that each thermal works
        """
        self._thermals = []
        inhibit_1 = numpy.array([70.0, 40.0, 0.0]) * 1000.0
        inhibit_2 = numpy.array([60.0, 60.0, 0.0]) * 1000.0
        inhibit_3 = numpy.array([65.0, 50.0, 0.0]) * 1000.0
        inhibit = (inhibit_1, inhibit_2, inhibit_3)
        inhibit = []
        for x_inhibit in inhibit:
            self._thermals.append(
                StochasticThermal(x_inhibit, 600, wscale, zi, P_work))
        while len(self._thermals) < n:
            candidate_X = numpy.random.rand(3) * x
            candidate_X[2] = 0.0

            P = 1.0 / (
                1.0 +
                numpy.exp(-1.0 * (self.w(candidate_X, zi) - 0.1)))

            Ptest = numpy.random.rand()
            inh = []
            for x_inhibit in inhibit:
                P_inhibit = 1.0 - numpy.exp(
                    -numpy.linalg.norm(candidate_X - x_inhibit) / 6.0e3)
                Ptest *= P_inhibit
            if Ptest > P:
                r = numpy.clip(
                    numpy.random.randn() * 100.0 + 600.0, 500.0, numpy.inf)
                w = numpy.clip(
                    numpy.random.randn() * 1.0 + wscale, 0.0, numpy.inf)
                w = wscale
                zi = numpy.clip(
                    numpy.random.randn() * sigma_zi + zi, 0.0, numpy.inf)

                if w == 0.0 or r == 0.0:
                    continue
                if P_work is None:
                    self._thermals.append(
                        Thermal(candidate_X, r, w, zi))
                else:
                    self._thermals.append(
                        StochasticThermal(candidate_X, r, w, zi, P_work))

        self._thermal_dict = {}
        for thermal in self._thermals:
            self._thermal_dict[thermal.id] = thermal

    def w(self, location, z):
        w = 0.0
        for t in self._thermals:
            w += t.w(location, z)

        return w

    def plot(self, show=True, save=True):
        """Plot the thermal field
        """
        x = numpy.vstack([therm.X for therm in self._thermals])
        plt.scatter(x[:,1], x[:,0])
        if save:
            f = plt.gcf()
            f.savefig('thermal_field.png', format='png', dpi=1000)
        if show:
            plt.show()

    def nearest(self, location, exclude=[]):
        """Get the nearest thermal

        Arguments:
            location: position to check
            exclude: exclude these thermals from nearest

        Returns:
            thermal: the thermal nearest this location
            idx: index of this thermal
        """
        r = numpy.fromiter(
            (t.distance(location) for t in self._thermals), dtype=float)
        idx = numpy.argmin(r)

        idx = 0
        sort_idx = numpy.argsort(r)
        while self._thermals[sort_idx[idx]].id in exclude:
            idx += 1
        return (self._thermals[sort_idx[idx]], sort_idx[idx])

        include = numpy.ones((len(self._thermals),), dtype=bool)
        include[exclude] = False

        idx = numpy.arange(len(self._thermals))[include]
        sub_idx = idx[numpy.argmin(r[include])]
        return (self._thermals[sub_idx], sub_idx)

    def thermals_in_range(
        self, location, range_limit, exclude=None):
        """Get all thermals within some distance of a point

        Arguments:
            location: the position to test
            range_limit: the distance to look for thermals within
            exclude: exclude these thermal indices

        Returns:
            to_thermals: vectors to those thermals
            thermals: the thermals in question sorted by range (closest
                first)
            idx: indices of those thermals
        """
        to_thermals = []
        distance = []
        thermals = []
        therm_idx = []
        for i, t in enumerate(self._thermals):
            v = t.vector_to(location)
            d = numpy.linalg.norm(v)
            if d < range_limit and i not in exclude:
                to_thermals.append(v)
                distance.append(d)
                thermals.append(t)
                therm_idx.append(i)

        to_thermals = numpy.array(to_thermals)
        distance = numpy.array(distance)

        sort_idx = numpy.argsort(distance)
        sorted_thermals = [thermals[idx] for idx in sort_idx]
        therm_idx = [therm_idx[idx] for idx in sort_idx]
        return (to_thermals[sort_idx], sorted_thermals, therm_idx)

    def reachable_thermals(self, amoeba, exclude=[]):
        """Get all thermals within a region which is reachable by an aircraft

        Arguments:
            amoeba: the glide amoeba, a shapely polygon
            exclude: exclude these thermal ids

        Returns:
            thermals: the thermals in no particular order
            idx: indices of these thermals
        """
        thermals = []
        therm_idx = []
        for i, thermal in enumerate(self._thermals):
            if amoeba.contains(thermal.point) and thermal.id not in exclude:
                thermals.append(thermal)
                therm_idx.append(i)

        return (thermals, therm_idx)
