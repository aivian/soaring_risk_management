import numpy

import scipy

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

        self._mc = 0.0
        self._n_minimum = 1
        self._P_landout_acceptable = 0.01

        self._pilot_characteristics = {
            'thermal_center_sigma': 600.0,
            'detection_range': 600.0
            }

        self._location = numpy.zeros((3,))

        self._state = 'optimize'

        self._speed_controllers = {
            'optimize': self._maccready_speed,
            'minimize_risk': self._range_speed,
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
            if self._location[-1] > 2000.0:
                self._visited_thermals.append(self._current_thermal.id)
            else:
                return

        destination, is_risk_tolerance_met = self.get_destination()
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
        self._current_thermal = found_thermal

    def get_destination(self):
        """Get a new destination

        Arguments:
            no arguments

        Returns:
            new_destination: numpy (3,) array giving xyz position of destination
            is_risk_tolerance_met: whether we can minimize risk as much as we
                want
        """
        next_turnpoint = self._task.next_turnpoint

        flight_path = numpy.vstack((self._location, next_turnpoint))

        #TODO: this needs a better speed thing, probably send it MC so that it
        # properly speeds up for the upwind legs
        glide_amoeba = self.glide_amoeba(self.select_speed())

        reachable_thermals = self._thermal_field.reachable_thermals(
            glide_amoeba)

        # assign the closest thermal to our course
        thermals = []
        P_landout = 1.0
        while (
            P_landout > self._P_landout_acceptable and
            len(reachable_thermals) > 0):
            ranges_to_course = numpy.fromiter(
                (numpy.linalg.norm(geometry.lines.point_line_distance(
                    thermal.X, flight_path)[0]) for
                thermal in reachable_thermals), dtype=float)

            thermals.append(
                reachable_thermals.pop(numpy.argmax(ranges_to_course)))
            P_work = numpy.fromiter(
                (thermal.P_work for thermal in thermals), dtype=float)
            P_landout = 1.0 - numpy.prod(1.0 - P_work)

        if len(reachable_thermals) == 0:
            is_risk_tolerance_met = False
        else:
            is_risk_tolerance_met = True

        # figure out which thermal we're targeting
        ranges_to_thermals = numpy.fromiter(
            (numpy.linalg.norm(thermal.X - self._location)
                for thermal in thermals), dtype=float)
        next_thermal = thermals[numpy.argmin(ranges_to_thermals)]

        if self._state is 'minimize_risk':
            next_thermal = self._thermal_field.get_nearest(self._location)

        destination = next_thermal.estimate_center(
            self._pilot_characteristics['thermal_center_sigma'])
        return (destination, is_risk_tolerance_met)

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
        v_candidate = self._maccready_speed
        while n_reachable < self._n_minimum:
            # do stuff to figure out how many we can reach
            v_candidate *= 0.99

            glide_amoeba = self.glide_amoeba(v_candidate)

            reachable_thermals = self._thermal_field.reachable_thermals(
                glide_amoeba)
            n_reachable = len(reachable_thermals)

        return v_candidate

    def detect_thermal(self):
        """Detect whether there's a thermal around

        Arguments:
            no arguments

        Returns:
            is_thermal_detected: bool indicates if we can detect a thermal
            thermal: the thermal we found. is None if we didn't find one
        """
        nearest_thermal = self._thermal_field.nearest_thermal(self._location)
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
        candidate_directions = numpy.arange(-numpy.pi, numpy.pi / 6, numpy.pi)
        for psi in candidate_directions:
            direction = numpy.array([cos(psi), sin(psi)])
            ground_speed = numpy.linalg.norm(direction * v + wind)
            glide_range = -ground_speed / sink_rate * self._location[2]

            reachable_set.append(direction * glide_range)

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
        self._destination = None
        self._thermal_center = None

    def set_destination(self, new_destination):
        """Give the navigator a new destination

        Arguments:
            new_destination: numpy (3,) array

        Returns:
            no returns
        """
        self._destination = new_destination
        self._thermal_center = None

    def thermal(self, thermal_location):
        """Tell the navigator to orbit a thermal

        Arguments:
            thermal_location: numpy (3,) givin the thermal position

        Returns:
            no returns
        """
        self._destination = None
        self._thermal_center = thermal_location

    def command(self, location):
        """Generate a steering command

        Arguments:

