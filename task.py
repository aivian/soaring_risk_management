import pdb

import numpy

class Turnpoint(object):
    """
    """
    def __init__(self, location, radius):
        """
        """
        self._location = location
        self._radius = radius

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

    def reached(self, location):
        """check if we've reached it
        """
        return self.distance(location) < self._radius

    @property
    def X(self):
        """Get the location
        """
        return self._location

    @property
    def point(self):
        """Get the location as a shapely point
        """
        return shapely.geometry.Point(self._x[0], self._x[1])


class SoaringTask(object):
    """A class to hold a sailplane task
    """
    def __init__(self, turnpoints, radii):
        """Constructor

        Arguments:
            waypoints: numpy (n,3) array giving turnpoint locations (m, NED)
            radii: numpy (n,) array giving acceptance radii for each turnpoint

        Returns:
            class instance
        """
        self._turnpoints = [Turnpoint(X, r) for X, r in zip(turnpoints, radii)]

        self._turnpoint_index = 0

    def update(self, location):
        """Update the task given the aircraft location

        advance the turnpoint if

        Arguments:
            location: numpy (3,) array giving the aircraft position

        Returns:
            task_completed: true if the task has been completed, false otherwise
        """
        if self.next_turnpoint.reached(location):
            self._turnpoint_index += 1

        if self._turnpoint_index >= len(self._turnpoints):
            return True

        return False

    def glide_ratio_to_next_waypoint(self, location):
        """Figure out the glide ratio to the next turnpoint

        Arguments:
            location: where the aircraft is

        Returns:
            glide_ratio: required to reach next
        """
        dx = self.next_turnpoint.distance(location)
        dz = location[2] - self.next_turnpoint.X[2]
        return -dx / dz

    @property
    def next_turnpoint(self):
        """Get the next turnpoint

        Arguments:
            no arguments

        Returns:
            next_turnpoint: the location of the next turnpoint
        """
        return self._turnpoints[self._turnpoint_index]

class AssignedTask(SoaringTask):
    """
    """
    def __init__(self, turnpoints, radii):
        """Constructor

        Arguments:
            waypoints: numpy (n,3) array giving turnpoint locations (m, NED)
            radii: numpy (n,) array giving acceptance radii for each turnpoint

        Returns:
            class instance
        """
        super(AssignedTask, self).__init__(turnpoints, radii)

