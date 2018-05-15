import pdb

import copy

import numpy
import scipy

import shapely.geometry

import task as task_module
import thermal_field as thermal_field_module

import geometry.conversions
import geodesy.conversions
import environments.earth
import robot_control.path_following

class SailplanePilot(object):
    """A decision-making class
    """
    def __init__(self, aircraft, thermal_field, task, characteristics={}):
        """Constructor

        Arguments:
            aircraft: definition of the aircraft we're flying
            thermal_field: the thermal environment this pilot is flying in
            task: where the aircraft is going

        Returns:
            class instance
        """
        self._navigator = Navigator()

        self._aircraft = aircraft
        self._thermal_field = thermal_field
        self._task = task

        self._visited_thermals = []
        self._non_working_thermals = []
        self._plan = []
        self._planned_thermals = []

        self._mc = 0.0
        self._P_err = 1.0

        self._characteristics = {
            'n_minimum': 1,
            'thermal_center_sigma': 400.0,
            'detection_range': 700.0,
            'P_landout_acceptable': 0.005,
            'final_glide_margin': 0.1,
            }

        for key, val in characteristics.iteritems():
            self._characteristics[key] = val

        self._location = numpy.zeros((3,))
        self._vehicle_state = numpy.zeros((5,))

        self._state = 'optimize'

        self._speed_controllers = {
            'prestart': self._range_speed,
            'optimize': self._maccready_speed,
            'minimize_risk': self._range_speed,
            'thermal': self._min_sink_speed,
            'final_glide': self._maccready_speed,
            }

    def update_vehicle_state(self, new_state, new_finite_state):
        """Get a new location

        Arguments:
            new_state: numpy array giving vehicle state
            new_finite_state: the current aircraft state

        Returns:
            no returns
        """
        self._location = new_state[:3]
        self._vehicle_state = new_state
        self._state = new_finite_state

    def leaving_thermal(self, thermal_id, working):
        """Leaving a thermal, remember about it
        """
        if not working and thermal_id not in self._non_working_thermals:
            self._non_working_thermals.append(thermal_id)

        if thermal_id not in self._visited_thermals:
            self._visited_thermals.append(thermal_id)

    def check_plan(self, plan=None, location=None):
        """
        """
        if plan is None:
            plan = self._plan
        if location is None:
            location = self._location

        if len(plan) == 0:
            return 1.0

        total_length = 0.0
        for idx, wp in enumerate(plan):
            if idx == 0:
                leg = wp.X - location
            else:
                leg = plan[idx-1].X - wp.X
            total_length += numpy.linalg.norm(
                leg * numpy.array([1.0, 1.0, 0.0]))

        P_work = numpy.fromiter(
            (thermal.P_work * self._P_err for thermal in plan), dtype=float)
        P_landout = numpy.prod(1.0 - P_work)
        return P_landout

    def set_plan(self, plan):
        """Set the plan for where to go

        Arguments:
            plan: the sequence of thermals to visit

        Returns:
            no returns
        """
        if len(plan) > 0:
            self._plan = plan
            self._planned_thermals = []
            for wp in self._plan:
                if isinstance(wp, thermal_field_module.Thermal):
                    self._planned_thermals.append(wp.id)

    def select_speed(self, w, state=None):
        """Figure out what speed to fly

        Arguments:
            no arguments

        Returns:
            v: speed to fly
        """
        if state is None:
            return self._speed_controllers[self._state](-w)
        else:
            return self._speed_controllers[state](-w)

    def _maccready_speed(self, w):
        """Get the MC speed

        Arguments:
            no arguments

        Returns:
            v: speed to fly (m/s)
        """
        return self._aircraft.speed_to_fly(Wm=w, MC=self._mc)

    def _range_speed(self, w):
        """Choose a speed to maintain connectedness

        Arguments:
            no arguments

        Returns:
            v: speed to fly (m/s)
        """
        return self._aircraft.speed_to_fly()

        n_reachable = 0
        v_candidate = self._maccready_speed()
        while n_reachable < self._characteristics['n_minimum']:
            # do stuff to figure out how many we can reach
            v_candidate *= 0.99

            glide_amoeba = self.glide_amoeba(v_candidate)

            reachable_thermals = self._thermal_field.reachable_thermals(
                glide_amoeba)
            n_reachable = len(reachable_thermals)

        return v_candidate

    def _min_sink_speed(self, w):
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
            self._characteristics['detection_range']):
            return (True, nearest_thermal)

        return (False, None)

    def glide_amoeba(self, v=None, vision_range=None):
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

            if vision_range is not None:
                reachable_set.append(
                    direction * vision_range + self._location[:2])
            else:
                reachable_set.append(
                    direction * glide_range + self._location[:2])

        reachable_set.append(reachable_set[0])
        amoeba = shapely.geometry.Polygon(reachable_set)
        return amoeba

    def set_mc(self, mc):
        """Set the maccready value
        """
        self._mc = mc

    @property
    def vehicle_state(self):
        """
        """
        return self._vehicle_state

    @property
    def risk_tolerance(self):
        """
        """
        return self._characteristics['P_landout_acceptable']

    @property
    def thermal_center_accuracy(self):
        """Get the sigma value for locating a thermal center from afar
        """
        return self._characteristics['thermal_center_sigma']

    @property
    def non_working_thermals(self):
        """Get the thermals that don't work
        """
        return self._non_working_thermals

    @property
    def visited_thermals(self):
        """Get the thermals that we've already visited
        """
        return self._visited_thermals

    def _navigate_to(self, destination, location):
        """Send a destination to the navigator

        Arguments:
            destination: some type of destination either a thermal or a
                turnpoint

        Returns:
            no returns
        """
        if isinstance(destination, thermal_field_module.Thermal):
            self._navigator.set_destination(
                destination.estimate_center(
                    self._characteristics['thermal_center_sigma']),
                location)
            return
        self._navigator.set_destination(destination.X, location)

    def navigate_plan(self, location):
        """set the destination from the plan
        """
        if len(self._plan) == 0:
            return
        if len(self._plan) == 1:
            self._navigate_to(self._plan[-1], location)
            return

        # figure out where turnpoints are in our plan, to make sure we don't
        # skip them...
        turnpoint_idx = -1
        idx = 0
        while idx < len(self._plan) and turnpoint_idx < 0:
            if isinstance(self._plan[idx], task_module.Turnpoint):
                turnpoint_idx = idx
            idx += 1

        if turnpoint_idx == 0:
            self._navigate_to(self._plan[0], location)
            return
        plan = numpy.vstack((wp.X for wp in self._plan[:turnpoint_idx]))

        # Check if we're within the waypoint acceptance distance of any points
        # if we are, then go to the next waypoint on the flight plan
        zero_z = numpy.array([1.0, 1.0, 0.0])
        distance = numpy.fromiter(
            (numpy.linalg.norm((wp - location) * zero_z) for wp in plan),
            dtype=float)
        if numpy.any(distance < self._navigator._acceptance_radius):
            # This requires that no two waypoints are located less than the
            # waypoint tolerance apart, or weird things will happen
            idx = numpy.argmin(distance) + 1
            # trap when we returned the last waypoint...
            if idx >= len(plan):
                idx = len(plan) - 1
            self._navigate_to(self._plan[idx], location)
            return

        # Find the closest point on the flight plan
        r, segment, idx = geometry.lines.point_line_distance(location, plan)

        # the computation returns the index of the beginning of the segment, so
        # increment it to get the waypoint of interest. Also Capture cases
        # where the last waypoint was returned (not possible I think, but lets
        # be safe)
        idx = idx + 1
        if idx >= len(plan):
            idx = len(plan) - 1

        # Check to see if the closest point on the flight plan is the first
        # waypoint. If it is, then we should go to it.
        if numpy.linalg.norm((r - plan[0]) * zero_z) < 1.0e-6:
            self._navigate_to(self._plan[0], location)
            return

        # Otherwise we just want to go to the end of the nearest segment
        self._navigate_to(self._plan[idx], location)
        return

    def thermal(self, new_thermal=None):
        """Go exploit a thermal
        """
        if new_thermal is None:
            new_thermal = self._current_thermal
        self._navigator.thermal(new_thermal.X)

    def phi_command(self, wind):
        """Get bank angle command

        Arguments:
            wind: numpy (3,) wind vector

        Returns:
            phi: bank angle command (Rad)
        """
        i_V = numpy.array([
            numpy.cos(self._vehicle_state[3]),
            numpy.sin(self._vehicle_state[3]),
            0.0]) * self._vehicle_state[4]
        inertial_direction = i_V / self._vehicle_state[4]
        acceleration_command = self._navigator.command(
            i_V, copy.deepcopy(self._vehicle_state[:3]))
        acceleration_normal_to_velocity = numpy.cross(
            numpy.cross(inertial_direction, acceleration_command),
            inertial_direction)
        acceleration_body_frame = geometry.rotations.zrot(
            self._vehicle_state[3]).dot(acceleration_normal_to_velocity)
        acceleration_body_frame += (
            geometry.helpers.unit_vector(2) * 9.806)
        phi_command = numpy.arctan2(
            acceleration_body_frame[1], acceleration_body_frame[2])
        phi_command = numpy.clip(phi_command, -numpy.pi / 3.0, numpy.pi / 3.0)
        return phi_command + numpy.deg2rad(numpy.random.randn() * 2.0)

    def theta_command(self, wind):
        """get pitch command

        Arguments:
            wind: numpy (3,) wind vector

        Returns:
            theta: pitch angle command (rad)
        """
        speed_command = self.select_speed(wind[2])
        speed_rate_command = numpy.clip(
            (speed_command - self._vehicle_state[4]) / 5.0,
            -3.0,
            3.0)
        theta_command = numpy.clip(
            numpy.arcsin(
                -speed_rate_command / environments.earth.constants['g0']),
            -numpy.pi / 12.0,
            numpy.pi / 6.0)
        return theta_command

    def check_thermal(self, thermal):
        """see if we should use a thermal
        """
        working = thermal.working(self._vehicle_state[:3])
        if thermal.id not in self._non_working_thermals and working:
            return True
        elif thermal.id not in self._non_working_thermals and working is False:
            self._non_working_thermals.append(thermal.id)
        return working

    def thermal_in_plan(self, thermal):
        """Check to see if a thermal is in our plan
        """
        return thermal.id in self._planned_thermals

    def on_final_glide(self, v=None, location=None):
        """Check if we can make a final glide

        Arguments:
            v: optional, the speed to check final glide at. If not specified
                then we'll just pick up the speed from the normal command
            location: optional location to check from. If not specified then
                check from where the aircraft is.

        Returns:
            on_final_glide: True if the aircraft can make final glide, false if
                it cannot
        """
        legs = self._task.legs_to_finish
        if location is None:
            location = self._vehicle_state[:3]
        legs = numpy.vstack((location, legs)) * numpy.array([1.0, 1.0, 0.0])
        distance = numpy.sum(numpy.linalg.norm(numpy.diff(legs, axis=0)))

        if v is None:
            v = self.select_speed(0.0, 'final_glide')

        time_to_go = distance / v
        time_to_fall = location[2] / self._aircraft.sink_rate(v)
        glide_factor = 1.0 + self._characteristics['final_glide_margin']
        if time_to_go * glide_factor < time_to_fall:
            return True
        return False

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
        self._L = 400.0
        self._R = 200.0
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
            robot_control.path_following.CirclingParkController(
                numpy.array(thermal_location, ndmin=2), self._R, self._L / 4.0)

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
                numpy.power(numpy.linalg.norm(velocity), 2.0) /
                environments.earth.constants['g0'] *
                numpy.cos(numpy.deg2rad(40.0)))

        zero_3 = numpy.array([1.0, 1.0, 0.0])
        self._controller.update_state(location * zero_3, velocity)
        return self._controller.command((True, False, True))

    def range_to_thermal(self, location):
        """
        """
        zero_3 = numpy.array([1.0, 1.0, 0.0])
        to_destination = numpy.linalg.norm(
            (self._thermal_center - location) * zero_3)
        return to_destination


