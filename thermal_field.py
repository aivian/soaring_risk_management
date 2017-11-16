import pdb

import numpy
import geometry.lines

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

        R = numpy.linalg.norm(self._x - xtest)
        w = numpy.exp(-R / self._r) * self._w
        w *= (numpy.cos(numpy.pi * ztest / self._zi / 4.0) + 1.0)
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
        return self._x - location

    @property
    def X(self):
        """Get the location
        """
        return self._x

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
    def __init__(self, x, r, w, zi, P):
        """Constructor

        Arguments:
            x: location (m, n/e)
            r: radius (m)
            w: updraft strength (m/s)
            zi: top (m)
            P: probability that this thermal works

        Returns:
            class instance
        """
        if numpy.random.rand() > P:
            w *= 0.0
        super(StochasticThermal, self).__init__(x, r, w, zi)

class ThermalField(object):
    def __init__(self, x, zi, sigma_zi, wscale, n, P=None):
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
        while len(self._thermals) < n:
            candidate_X = numpy.random.rand(2) * x

            P = 1.0 / (
                1.0 +
                numpy.exp(-1.0 * (self.w(candidate_X, zi) - 0.1)))
            Ptest = numpy.random.rand()
            if Ptest > P:
                r = numpy.clip(
                    numpy.random.randn() * 100.0 + 250.0, 0.0, numpy.inf)
                w = numpy.clip(
                    numpy.random.randn() * 1.0 + wscale, 0.0, numpy.inf)
                zi = numpy.clip(
                    numpy.random.randn() * sigma_zi + zi, 0.0, numpy.inf)

                if w == 0.0 or r == 0.0:
                    continue
                if P is None:
                    self._thermals.append(
                        Thermal(candidate_X, r, w, zi))
                else:
                    self._thermals.append(
                        StochasticThermal(candidate_X, r, w, zi, P))

    def w(self, location, z):
        w = 0.0
        for t in self._thermals:
            w += t.w(location, z)

        return w

    def plot(self, show=True):
        """Plot the thermal field
        """
        x = numpy.vstack([therm._x for therm in self._thermals])
        plt.scatter(x[:,0], x[:,1])
        if show:
            plt.show()

    def get_nearest(self, location, exclude=None):
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
        if exclude is None:
            idx = numpy.argmin(r)
            return (self._thermals[idx], idx)

        include = numpy.ones((len(self._thermals),), dtype=bool)
        include[exclude] = False

        idx = numpy.arange(len(self._thermals))[include]
        sub_idx = idx[numpy.argmin(r[include])]
        return (self._thermals[sub_idx], sub_idx)

    def get_thermals_in_range(
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
