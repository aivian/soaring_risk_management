"""This is to implement a sailplane model
"""
import pdb

import numpy
import scipy

import environments.earth

import simulation.dynamic_system

class SailplaneSimulation(simulation.dynamic_system.DynamicSystem):
    """A class for sailplane simulation of a sailplane

    State:
        N: north position (m)
        E: east posotion (m)
        D: down position (m)
        psi: heading (rad)
        va: indicated airspeed (m/s)
    """
    def __init__(
        self,
        polar,
        parameters,
        initial_state,
        state_record,
        simulation_parameters):
        """Constructor

        Arguments:
            polar: an instance of SailplanePerformance
            parameters: a dictionary with fields:
                Sref: reference wing area (m^2)
                mass: aircraft mass (kg)
            initial_state: state vector specifying the initial aircraft state:
                N: north position (m)
                E: east posotion (m)
                D: down position (m)
                psi: heading (rad)
                va: indicated airspeed (m/s)
            state_record: class which will record the state and input
            simulation_parameters: dictionary of sim setup
                dt: timestep size

        Returns:
            class instance
        """
        self._dt = simulation_parameters['dt']
        super(SailplaneSimulation, self).__init__(
            initial_state, self._dt, self._sailplane_dynamics)

        self._polar = polar
        self._atmosphere = environments.earth.Atmosphere()

        self._Sref = parameters['Sref']
        self._mass = parameters['mass']

        self._state_record = state_record

        self.initialize(0.0, initial_state)

    def initialize(self, t, X):
        """Initialize the aircraft

        Arguments:
            t: start time (s)
            X: initial state
                N: north position (m)
                E: east posotion (m)
                D: down position (m)
                psi: heading (rad)
                va: indicated airspeed (m/s)

        Returns:
            no returns
        """
        self._X = X
        self._t = t
        self._state_record.reset(self._t, self._X)

    def step(self, t, phi, theta, wind):
        """Step the simulation

        Arguments:
            phi: bank angle (rad)
            theta: pitch angle (sort of. this is the pitch angle relative
                to that required to maintain speed. We approximate alpha as
                0 for this) (rad)
            wind: numpy (3,) array
                north wind (m/s)
                east wind (m/s)
                down wind (m/s)

        Returns:
            no returns
        """
        sigma = (
            self._atmosphere.rho_isa(-self._X[2]) /
            self._atmosphere.rho_isa(0.0))
        U_pilot = numpy.array([
            self._atmosphere.g() * numpy.tan(phi),
            -theta / self._X[4] / numpy.sqrt(sigma)])
        U = numpy.hstack((U_pilot, wind))
        self.rk4(U)
        self._state_record.update_state(t, self._X, U)

    def _sailplane_dynamics(self, X, U):
        """Dynamics function

        Arguments:
            X: state
                N: north position (m)
                E: east posotion (m)
                D: down position (m)
                psi: heading (rad)
                va: indicated airspeed (m/s)
            U: input
                b_ay: body axis y-acceleration (m/s/s)
                va_dot: airspeed rate (m/s/s)
                wn: north wind (m/s)
                we: east wind (m/s)
                wd: down wind (m/s)

        Returns:
            Xdot: state rate
                vn: north velocity (m/s)
                ve: east velocity (m/s)
                vd: down velocity (m/s)
                psi_dot: heading rate (rad/s)
                va_dot: indicated airspeed rate (m/s/s)
        """
        sigma = (
            self._atmosphere.rho_isa(-X[2]) /
            self._atmosphere.rho_isa(0.0))
        tas = X[4] / numpy.sqrt(sigma)

        v_n = numpy.cos(X[3]) * tas + U[2]
        v_e = numpy.sin(X[3]) * tas + U[3]
        v_d = -self._polar.sink_rate(X[4]) / numpy.sqrt(sigma) + U[4]
        psi_dot = U[0] / tas
        va_dot = U[1]
        return numpy.array([v_n, v_e, v_d, psi_dot, va_dot])

class QuadraticPolar(object):
    """
    """
    def __init__(self, polar_coeffs, v0, v1, ref_mass, mass):
        """ Constructor
        """
        super(QuadraticPolar, self).__init__()
        self.ref_coeffs = polar_coeffs
        self.v_range = (v0, v1)
        self.ref_mass = ref_mass
        self.set_mass(mass)

    def speed_to_fly(self, Wm=0.0, MC=0.0, V_h=0.0):
        """Compute traditional steady-state speed to fly by factoring in the
        current airmass motion, the expected climb rate in the next thermal
        (ring setting), and the current headwind value. This differes slightly
        from some derivations of speed to fly in the headwind term. This
        derivation assumes that thermals have a fixed ground location and thus
        ground speed is important rather than just airspeed.

        Arguments:
            Wm: vertical air motion (netto), positive upwards (m/s)
            MC: expected climb rate in next thermal (MC number)
            V_h: headwind (positive opposing flight direction)

        Returns:
            (speed to fly, optimal cruise speed)
        """
        MC = max(MC, 0.0) # Cl is always positive (pg. 106 Reichmann)
        # Nate's derivation, see dissertation doc (this formatting is a mess)
        V_stf = V_h - numpy.sqrt(max(self.polar[0] * (
            numpy.power(self.polar[0] * V_h, 2.0) +
            self.polar[1] * V_h +
            self.polar[2] +
            Wm -
            MC), 0.0)) / self.polar[0]

        return V_stf

    def min_sink_speed(self):
        """Compute minimum sink airspeed. This is the "top" of the quadratic
        speed polar.

        Arguments:
            no arguments

        Returns:
            (IAS for min sink, sink rate)
        """
        V_min_sink = -1.0*self.polar[1] / (2.0 * self.polar[0])

        return V_min_sink

    def set_mass(self, mass):
        """ Set a mass value

        Arguments:
            mass: the true mass as flown

        Returns:
            no returns, but will adjust the polar variable
        """
        self._mass = mass
        mass_ratio = numpy.sqrt(self._mass / self.ref_mass)
        self.polar = (
            self.ref_coeffs[0] / mass_ratio,
            self.ref_coeffs[1],
            self.ref_coeffs[2] * mass_ratio)

    def sink_rate(self, v, n=None):
        """Compute the sink rate

        Arguments:
            v: airspeed to compute sink rate for (m/s)
            n: load factor, optional (defaults to 1.0)

        Returns:
            w: sink rate at that speed
        """
        polar = numpy.array(self.polar)
        if n is not None:
            n_ratio = numpy.sqrt(n)
            polar *= numpy.array([1 / n_ratio, 1.0, n_ratio])
        return numpy.polyval(polar, v)

    @property
    def min_sink(self):
        """Returns the minimum sink rate of the aircraft. Note that this value
        is always negative for non-magical aircraft."""
        return self.polar[2] - (self.polar[1]**2.0) / (4.0 * self.polar[0])

    def v_bar(self, w_thermal):
        """Compute average speed for a given thermal strength

        Arguments:
            w_thermal: thermal updraft strength

        Returns:
            v_bar: average cross-country speed
        """
        v = self.speed_to_fly(0.0, w_thermal)
        scale_factor = -(w_thermal /
            (-w_thermal + self.sink_rate(v)))
        return v * scale_factor
