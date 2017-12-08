import pdb

import numpy

import scipy

import shapely.geometry

import geometry.conversions
import geodesy.conversions
import environments.earth
import robot_control.path_following

class SailplanePilot(object):
    """A decision-making class
    """
    def __init__(self, aircraft, thermal_field, task):
        """Constructor

        Arguments:
            aircraft: definition of the aircraft we're flying
            thermal_field: the thermal environment this pilot is flying in
            task: where the aircraft is going

        Returns:
            class instance
        """
        self._aircraft = aircraft
        self._thermal_field = thermal_field
        self._task = task

        self._visited_thermals = []
        self._plan = []

        self._mc = 0.0
        self._n_minimum = 1
        self._P_landout_acceptable = 0.01

        self._pilot_characteristics = {
            'thermal_center_sigma': 600.0,
            'detection_range': 600.0
            }

        self._location = numpy.zeros((3,))
        self._navigator = Navigator()

        self._state = 'optimize'

        self._speed_controllers = {
            'optimize': self._maccready_speed,
            'minimize_risk': self._range_speed,
            'thermal': self._min_sink_speed,
            }

    def update_location(self, new_location, evaluate_state=True):
        """Get a new location

        Arguments:
            new_location: numpy (3,) giving xyz position (m)
            evaluate_state: optional, should we evaluate our state defaults True

        Returns:
            no returns
        """
        self._location = new_location

        if evaluate_state:
            self.evaluate_state()

    def evaluate_state(self):
        """evaluate which state we're in

        Arguments:
            no arguments

        Returns:
            no returns
        """
        if self._state is 'thermal':
            # figure out if we should leave
            if -self._location[-1] > self._current_thermal.zmax * 0.99:
                self._visited_thermals.append(self._current_thermal.id)
            else:
                return

        # we check through the point before end of the plan here because the
        # last element is the next turnpoint we're going to and it doesn't have
        # a probability of working...
        is_risk_tolerance_met = (
            self.check_plan(self._plan[:-1], self._location)
            < self._P_landout_acceptable)
        if is_risk_tolerance_met:
            self._state = 'optimize'
        else:
            self._state = 'minimize_risk'

        found_thermal, thermal = self.detect_thermal()
        if not found_thermal:
            return
        if thermal.id in self._visited_thermals and self._state is 'optimize':
            return

        self._state = 'thermal'
        self._current_thermal = thermal

    def check_plan(self, plan, location):
        """
        """
        if len(plan) == 0:
            return 1.0

        if self._task.glide_ratio_to_next_waypoint(location) < 10.0:
            return 0.0

        P_work = numpy.fromiter(
            (thermal.P_work for thermal in plan), dtype=float)
        P_landout = numpy.prod(1.0 - P_work)
        return P_landout

    def plan_destination(self):
        """Get a new destination

        Arguments:
            no arguments

        Returns:
            new_destination: numpy (3,) array giving xyz position of destination
            is_risk_tolerance_met: whether risk is as low as desired
            plan: the plan we're gonna fly (thermals)
        """
        next_turnpoint = self._task.next_turnpoint

        wind = numpy.zeros((3,))
        airspeed = self.select_speed()
        sink_rate = self._aircraft.sink_rate(airspeed)
        glide_direction = next_turnpoint.X - self._location
        glide_direction /= numpy.linalg.norm(
            glide_direction * numpy.array([1.0, 1.0, 0.0]))

        ground_speed = numpy.linalg.norm(glide_direction * airspeed + wind)
        glide_range = ground_speed / sink_rate * self._location[2]

        flight_path = numpy.vstack((
            self._location,
            self._location + glide_direction * glide_range))

        if self._task.glide_ratio_to_next_waypoint(self._location) < 10.0:
            plan = [self._task.next_turnpoint,]
            return (plan[0].X, True, plan)

        #TODO: this needs a better speed thing, probably send it MC so that it
        # properly speeds up for the upwind legs
        glide_amoeba = self.glide_amoeba(airspeed)

        reachable_thermals = self._thermal_field.reachable_thermals(
            glide_amoeba)[0]

        # assign the closest thermal to our course
        thermals = []
        P_landout = 1.0
        zero_3 = numpy.array([1.0, 1.0, 0.0])
        while (
            P_landout > self._P_landout_acceptable and
            len(reachable_thermals) > 0):
            leg_lengths = numpy.linalg.norm(
                numpy.diff(flight_path, axis=0), axis=1)
            idx_to_expand = numpy.argmax(leg_lengths)
            leg_to_expand = flight_path[idx_to_expand:idx_to_expand + 2]
            ranges_to_course = numpy.fromiter(
                (numpy.linalg.norm(geometry.lines.point_line_distance(
                    thermal.X * zero_3, leg_to_expand * zero_3)[0]) for
                thermal in reachable_thermals), dtype=float)

            thermals.insert(
                idx_to_expand,
                reachable_thermals.pop(numpy.argmin(ranges_to_course)))
            flight_path = numpy.vstack((
                flight_path[:idx_to_expand + 1],
                thermals[idx_to_expand].X,
                flight_path[idx_to_expand + 1:]))
            P_landout = self.check_plan(thermals, self._location)

        if len(reachable_thermals) == 0:
            is_risk_tolerance_met = False
        else:
            is_risk_tolerance_met = True

        if len(thermals) == 0 or self._state is 'minimize_risk':
            next_thermal = self._thermal_field.nearest(self._location)[0]
        else:
            # figure out which thermal we're targeting
            ranges_to_thermals = numpy.fromiter(
                (numpy.linalg.norm(thermal.X - self._location)
                    for thermal in thermals), dtype=float)
            next_thermal = thermals[numpy.argmin(ranges_to_thermals)]

        destination = next_thermal.estimate_center(
            self._pilot_characteristics['thermal_center_sigma'])

        plan = thermals + [self._task.next_turnpoint]
        return (destination, is_risk_tolerance_met, plan)

    def set_plan(self, plan):
        """Set the plan for where to go

        Arguments:
            plan: the sequence of thermals to visit

        Returns:
            no returns
        """
        if len(plan) > 0:
            self._plan = plan

    def select_speed(self):
        """Figure out what speed to fly

        Arguments:
            no arguments

        Returns:
            v: speed to fly
        """
        v = self._speed_controllers[self._state]()
        return v

    def _maccready_speed(self):
        """Get the MC speed

        Arguments:
            no arguments

        Returns:
            v: speed to fly (m/s)
        """
        return self._aircraft.speed_to_fly(self._mc)

    def _range_speed(self):
        """Choose a speed to maintain connectedness

        Arguments:
            no arguments

        Returns:
            v: speed to fly (m/s)
        """
        n_reachable = 0
        v_candidate = self._maccready_speed()
        while n_reachable < self._n_minimum:
            # do stuff to figure out how many we can reach
            v_candidate *= 0.99

            glide_amoeba = self.glide_amoeba(v_candidate)

            reachable_thermals = self._thermal_field.reachable_thermals(
                glide_amoeba)
            n_reachable = len(reachable_thermals)

        return v_candidate

    def _min_sink_speed(self):
        return self._aircraft.min_sink_speed()

    def detect_thermal(self):
        """Detect whether there's a thermal around

        Arguments:
            no arguments

        Returns:
            is_thermal_detected: bool indicates if we can detect a thermal
            thermal: the thermal we found. is None if we didn't find one
        """
        nearest_thermal = self._thermal_field.nearest(self._location)[0]
        if (
            nearest_thermal.distance(self._location) <
            self._pilot_characteristics['detection_range']):
            return (True, nearest_thermal)

        return (False, None)

    def glide_amoeba(self, v):
        """Figure out where we can go

        Arguments:
            v: speed

        Returns:
            amoeba: the amoeba we can get to
        """
        wind = numpy.zeros((2,))
        sink_rate = self._aircraft.sink_rate(v)
        reachable_set = []
        candidate_directions = numpy.arange(-numpy.pi, numpy.pi, numpy.pi / 6)
        for psi in candidate_directions:
            direction = numpy.array([numpy.cos(psi), numpy.sin(psi)])
            ground_speed = numpy.linalg.norm(direction * v + wind)
            glide_range = ground_speed / sink_rate * self._location[2]

            reachable_set.append(direction * glide_range + self._location[:2])

        reachable_set.append(reachable_set[0])
        amoeba = shapely.geometry.Polygon(reachable_set)
        return amoeba

    @property
    def state(self):
        """Get the current state
        """
        return self._state

class Navigator(object):
    """A class to handle navigating this sailplane
    """
    def __init__(self):
        """Constructor

        Arguments:
            no arguments

        Returns:
            no returns
        """
        self._L = 200.0
        self._R = 0.0
        self._destination = None
        self._thermal_center = None
        self._controller = None
        self._plan = numpy.array([], ndmin=2)
        self._acceptance_radius = 50.0

    def set_destination(self, new_destination, location):
        """Give the navigator a new destination

        Arguments:
            new_destination: numpy (3,) array
            location: where the aircraft is right now

        Returns:
            no returns
        """
        self._destination = new_destination
        self._thermal_center = None
        self._controller = robot_control.path_following.ParkController(
            numpy.vstack((location, new_destination)), self._L)

    def set_plan(self, new_plan):
        """Give the navigator a new plan

        Arguments:
            new_plan: a sequence of thermals to visit

        Returns:
            no returns
        """
        self._plan = numpy.vstack((wp.X for wp in new_plan))

    def destination_from_plan(self, location):
        """set the destination from the plan
        """
        if self._plan.shape[0] == 0:
            return numpy.zeros((3,))
        if self._plan.shape[0] == 1:
            return self._plan[0]

        # Check if we're within the waypoint acceptance distance of any points
        # if we are, then go to the next waypoint on the flight plan
        zero_z = numpy.array([1.0, 1.0, 0.0])
        distance = numpy.fromiter(
            (numpy.linalg.norm((wp - location) * zero_z) for wp in self._plan),
            dtype=float)
        if numpy.any(distance < self._acceptance_radius):
            # This requires that no two waypoints are located less than the
            # waypoint tolerance apart, or weird things will happen
            idx = numpy.argmin(distance) + 1
            # trap when we returned the last waypoint...
            if idx >= len(self._plan):
                idx = len(self._plan) - 1
            destination = self._plan[idx]
            return destination

        # Find the closest point on the flight plan
        r, segment, idx = geometry.lines.point_line_distance(
            location, self._plan)

        # the computation returns the index of the beginning of the segment, so
        # increment it to get the waypoint of interest. Also Capture cases
        # where the last waypoint was returned (not possible I think, but lets
        # be safe)
        idx = idx + 1
        if idx >= len(self._plan):
            idx = len(self._plan) - 1

        # Check to see if the closest point on the flight plan is the first
        # waypoint. If it is, then we should go to it.
        if numpy.linalg.norm((r - self._plan[0]) * zero_z) < 1.0e-6:
            destination = self._plan[0]
            return destination

        # Otherwise we just want to go to the end of the nearest segment
        destination = self._plan[idx]
        return destination


    def thermal(self, thermal_location):
        """Tell the navigator to orbit a thermal

        Arguments:
            thermal_location: numpy (3,) giving the thermal position

        Returns:
            no returns
        """
        self._destination = None
        self._thermal_center = thermal_location
        self._controller =\
            robot_control.path_following.CirclingParkContoller(
                numpy.array(thermal_location, ndmin=2), self._R, self._L)

    def command(self, velocity, location):
        """Generate a steering command

        Arguments:
            velocity: the aircraft velocity vector
            location: where the aircraft is right now

        Returns:
            i_a: inertial acceleration command (m/s/s)
        """
        if self._controller is None:
            return numpy.zeros((3,))

        if self._thermal_center is not None:
            self._controller.R = (
                numpy.power(numpy.linalg.norm(velocity, 2.0)) /
                environments.earth.constants['g0'] *
                numpy.cos(numpy.deg2rad(40.0)))

        self._controller.update_state(location, velocity)
        return self._controller.command()
