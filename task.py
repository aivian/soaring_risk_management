import pdb

import numpy
import geometry.lines

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

    @property
    def P_work(self):
        """
        """
        return 0.0

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

        self._started = False
        self._finished = False
        self.turnpoint_reached = False

    def update(self, location):
        """Update the task given the aircraft location

        advance the turnpoint if

        Arguments:
            location: numpy (3,) array giving the aircraft position

        Returns:
            task_completed: true if the task has been completed, false otherwise
            turnpoint_reached: true if we've reached a turnpoint
        """
        if self._finished:
            return (self._finished, True)

        self.turnpoint_reached = False
        if self.next_turnpoint.reached(location):
            if self._turnpoint_index == 0:
                self._started = True
            self._turnpoint_index += 1
            self.turnpoint_reached = True

        if self._turnpoint_index >= len(self._turnpoints):
            self._finished = True
            self.turnpoint_reached = True
            return (self._finished, self.turnpoint_reached)

        return (self._finished, self.turnpoint_reached)

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

    def get_turnpoint(self, idx=None, relative_idx=None):
        """Get a turnpoint

        Arguments:
            idx: optional, the turnpoint we want
            relative_idx: optional, the turnpoint we want relative to the
                current turnpoint

        Returns:
            turnpoint
        """
        if idx is not None:
            idx = numpy.clip(idx, 0, len(self._turnpoints) - 1)
        if relative_idx is not None:
            idx = numpy.clip(
                self._turnpoint_index + relative_idx,
                0,
                len(self._turnpoints) - 1)
        return self._turnpoints[idx]

    @property
    def next_turnpoint(self):
        """Get the next turnpoint

        Arguments:
            no arguments

        Returns:
            next_turnpoint: the location of the next turnpoint
        """
        if self._turnpoint_index >= len(self._turnpoints):
            return 0
        return self._turnpoints[self._turnpoint_index]

    @property
    def next_leg(self):
        """Get the next leg vector

        Arguments:
            no arguments

        Returns:
            next_leg: numpy (2,3) array giving the next leg (after reaching the
                next waypoint). Returns None if we're on the last leg
        """
        if self._turnpoint_index >= (len(self._turnpoints) - 1):
            return None
        start_wp = self._turnpoints[self._turnpoint_index].X
        end_wp = self._turnpoints[self._turnpoint_index + 1].X
        return numpy.vstack((start_wp, end_wp))

    @property
    def started(self):
        """
        """
        return self._started

    @property
    def finished(self):
        """
        """
        return self._finished

    def progress(self, location):
        """Figure out how much progress we've made on the task

        Arguments:
            location: numpy (3,) array giving aircraft position N/E/D

        Returns:
            progress: how many km has been completed
        """
        if self.finished:
            return self.distance()
        if not self.started:
            return 0.0
        points = numpy.vstack((tp.X for tp in self._turnpoints))
        next_leg = points[self._turnpoint_index - 1:self._turnpoint_index + 1]
        legs = numpy.diff(points, axis=0)
        leg_distances = numpy.linalg.norm(legs, axis=1)
        basic_distance = numpy.sum(leg_distances[:self._turnpoint_index - 1])
        leg_progress = numpy.linalg.norm(
            geometry.lines.point_line_distance(location, next_leg)[0] +
            location -
            self._turnpoints[self._turnpoint_index - 1] .X)
        return basic_distance + leg_progress

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

    def distance(self):
        """Get the task distance
        """
        X_points = numpy.vstack((pt.X for pt in self._turnpoints))
        delta = numpy.diff(X_points, axis=0)
        distance = numpy.linalg.norm(delta, axis=1)
        return numpy.sum(distance)

    @property
    def legs_to_finish(self):
        """Get the legs required in order to finish the task

        Arguments:
            no arguments

        Returns:
            legs: numpy (n,3) array giving the legs which must be completed
                in order to finish the task
        """
        tp_list = []
        for tp in self._turnpoints[self._turnpoint_index:]:
            tp_list.append(tp.X)

        return numpy.vstack(tp_list)
