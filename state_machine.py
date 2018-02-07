import pdb
import numpy
import copy
import itertools

import geometry.lines
import geometry.conversions

import task as task_module

def create_pilot_machine(thermal_field, pilot, navigator, aircraft, task):
    """
    """
    pilot_machine = StateMachine()

    pre_start_state = PreStart(
        'prestart',
        thermal_field,
        pilot,
        navigator,
        aircraft,
        task)
    pre_start_transitions = {
        'started': 'optimize',
        'thermal': 'thermal'}

    optimize_state = Optimize(
        'optimize',
        thermal_field,
        pilot,
        navigator,
        aircraft,
        task)
    optimize_transitions = {
        'final_glide': 'final_glide',
        'replan': 'optimize',
        'minimize_risk': 'minimize_risk',
        'thermal': 'thermal'}

    minimize_risk_state = MinimizeRisk(
        'minimize_risk',
        thermal_field,
        pilot,
        navigator,
        aircraft,
        task)
    minimize_risk_transitions = {
        'final_glide': 'final_glide',
        'replan': 'minimize_risk',
        'survive': 'survive',
        'thermal': 'thermal',
        'optimize': 'optimize'}

    survive_state = Survive(
        'survive',
        thermal_field,
        pilot,
        navigator,
        aircraft,
        task)
    survive_transitions = {
        'final_glide': 'final_glide',
        'replan': 'survive',
        'thermal': 'thermal'}

    thermal_state = Thermal(
        'thermal',
        thermal_field,
        pilot,
        navigator,
        aircraft,
        task)
    thermal_transitions = {
        'final_glide': 'final_glide',
        'optimize': 'optimize'}

    final_glide_state = FinalGlide(
        'final_glide',
        thermal_field,
        pilot,
        navigator,
        aircraft,
        task)
    final_glide_transitions = {
        'replan': 'final_glide',
        'below_glide': 'optimize'}

    pilot_machine.add_state(pre_start_state)
    pilot_machine.add_transition_map(pre_start_state, pre_start_transitions)
    pilot_machine.add_state(optimize_state)
    pilot_machine.add_transition_map(optimize_state, optimize_transitions)
    pilot_machine.add_state(minimize_risk_state)
    pilot_machine.add_transition_map(
        minimize_risk_state, minimize_risk_transitions)
    pilot_machine.add_state(survive_state)
    pilot_machine.add_transition_map(survive_state, survive_transitions)
    pilot_machine.add_state(thermal_state)
    pilot_machine.add_transition_map(thermal_state, thermal_transitions)
    pilot_machine.add_state(final_glide_state)
    pilot_machine.add_transition_map(final_glide_state, final_glide_transitions)

    return pilot_machine

def create_optimize_machine(thermal_field, pilot, navigator, aircraft, task):
    """
    """
    pilot_machine = StateMachine()

    pre_start_state = PreStart(
        'prestart',
        thermal_field,
        pilot,
        navigator,
        aircraft,
        task)
    pre_start_transitions = {
        'started': 'optimize',
        'thermal': 'thermal'}

    optimize_state = Optimize(
        'optimize',
        thermal_field,
        pilot,
        navigator,
        aircraft,
        task)
    optimize_transitions = {
        'final_glide': 'final_glide',
        'minimize_risk': None,
        'replan': 'optimize',
        'thermal': 'thermal'}

    thermal_state = Thermal(
        'thermal',
        thermal_field,
        pilot,
        navigator,
        aircraft,
        task)
    thermal_transitions = {
        'final_glide': 'final_glide',
        'optimize': 'optimize'}

    final_glide_state = FinalGlide(
        'final_glide',
        thermal_field,
        pilot,
        navigator,
        aircraft,
        task)
    final_glide_transitions = {
        'replan': 'final_glide',
        'below_glide': 'optimize'}

    pilot_machine.add_state(pre_start_state)
    pilot_machine.add_transition_map(pre_start_state, pre_start_transitions)
    pilot_machine.add_state(optimize_state)
    pilot_machine.add_transition_map(optimize_state, optimize_transitions)
    pilot_machine.add_state(thermal_state)
    pilot_machine.add_transition_map(thermal_state, thermal_transitions)
    pilot_machine.add_state(final_glide_state)
    pilot_machine.add_transition_map(final_glide_state, final_glide_transitions)

    return pilot_machine

class StateMachine(object):
    """
    """
    def __init__(self):
        """Create a state machine

        Arguments:
            no arguments

        Returns:
            no returns
        """
        self._states = {}
        self._transitions = {}
        self._current_state = None

    def add_state(self, new_state):
        """Add a state to the machine

        Arguments:
            new_state: a BaseState instance to add

        Returns:
            no returns
        """
        assert isinstance(new_state, BaseState),\
            'new_state must be a BaseState instance'

        self._states[new_state.name] = new_state
        self._transitions[new_state.name] = {}

    def add_transition(self, from_state, transition_id, to_state):
        """Add a transition

        Arguments:
            from_state: the name of the state starting this transition
            transition_id: the name of the transition
            to_state: the name of the state we'd like to end up in

        Returns:
            no returns
        """
        if isinstance(from_state, BaseState):
            from_state = from_state.name

        if isinstance(to_state, BaseState):
            to_state = to_state.name

        assert from_state in self._states,\
            'state "{}" is not in the state machine'.format(from_state)
        assert to_state in self._states,\
            'state "{}" is not in the state machine'.format(to_state)

        self._transitions[from_state][transition_id] = to_state
        self._states[from_state].register_transition(transition_id)

    def add_transition_map(self, from_state, transition_map, append=True):
        """Add a transition map, defining the transitions allowed from a state

        Arguments:
            from_state: the name of the state starting this transition
            transition_map: dictionary defining transition names and the states
                they transition to
            append: should the new map be appended to or replace the current
                transition map for this state. Defaults True

        Returns:
            no returns
        """
        if isinstance(from_state, BaseState):
            from_state = from_state.name
        assert from_state in self._states,\
            'state "{}" is not in the state machine'.format(from_state)

        if not append:
            self._transitions[from_state] = {}

        for transition_id in transition_map.iterkeys():
            self._states[from_state].register_transition(transition_id)
        self._transitions[from_state].update(transition_map)

    def set_state(self, new_state):
        """
        """
        assert new_state in self._states, 'new_state, {}, not found in states'
        self._current_state = new_state

    def execute(self):
        """Run the machine

        This function should be called whenever the state machine should check
        for a transition

        Arguments:
            no arguments

        Returns:
            state: the state the machine is currently in
            transition: the transition state of the machine, None if no
                transition occurs
        """
        if self._current_state is None:
            return None, None

        transition = self._states[self._current_state].execute()
        if transition is not None:
            assert transition in self._transitions[self._current_state],\
                'transition, {} not an allowed transition for state, {}'.format(
                    transition, self._current_state)
            if self._transitions[self._current_state][transition] is None:
                return self._current_state, None
            self._states[self._current_state].exit_actions()
            self._current_state =\
                self._transitions[self._current_state][transition]

        return self._current_state, transition

    @property
    def state(self):
        """Get the current state
        """
        return self._current_state

class BaseState(object):
    """
    """
    def __init__(self, name):
        """
        """
        super(BaseState, self).__init__()

        self._name = name
        self._transitions = []

        self._active = False

    def entry_actions(self):
        """
        """
        self._active = True

    def exit_actions(self):
        """
        """
        self._active = False

    def recurrent_actions(self):
        """Do anything which should be done repeatedly
        """
        return

    def execute(self):
        """Main callback
        """
        if not self._active:
            self.entry_actions()

        self.recurrent_actions()

        transition = self._check_transition_criteria()

        if not transition:
            return None

        assert transition in self._transitions,\
            'transition {} not registered for this state'.format(transition)

        return transition

    def register_transition(self, new_transition):
        """Add a valid transition to this state

        Arguments:
            new_transition: the name of a transition to add

        Returns:
            no returns
        """
        self._transitions.append(new_transition)

    def _check_transition_criteria(self):
        """
        """
        return None

    @property
    def name(self):
        """Get the name
        """
        return self._name

class SoaringState(BaseState):
    """
    """
    def __init__(
        self,
        name,
        thermal_field,
        pilot,
        navigator,
        aircraft,
        task):
        """
        """
        super(SoaringState, self).__init__(name)

        self._thermal_field = thermal_field
        self._pilot = pilot
        self._navigator = navigator
        self._aircraft = aircraft
        self._task = task

    def _check_transition_criteria(self):
        """
        """
        if self._pilot.vehicle_state[2] > 0:
            return 'landed'

        if self._task.finished:
            return 'finished'

        return None

class PreStart(SoaringState):
    """
    """
    def __init__(
        self,
        name,
        thermal_field,
        pilot,
        navigator,
        aircraft,
        task):
        """
        """
        super(PreStart, self).__init__(
            name,
            thermal_field,
            pilot,
            navigator,
            aircraft,
            task)

    def entry_actions(self):
        """
        """
        super(PreStart, self).entry_actions()
        self._navigator.set_destination(
            self._task.next_turnpoint.X, self._pilot.vehicle_state[:3])

    def recurrent_actions(self):
        self._pilot.navigate_plan(self._pilot.vehicle_state[:3])

    def _check_transition_criteria(self):
        """
        """
        transition = super(PreStart, self)._check_transition_criteria()
        if transition:
            return transition

        if self._task.started:
            return 'started'

        if self._pilot.detect_thermal()[0]:
            return 'thermal'

        return None

class Optimize(SoaringState):
    """
    """
    def __init__(
        self,
        name,
        thermal_field,
        pilot,
        navigator,
        aircraft,
        task):
        """
        """
        super(Optimize, self).__init__(
            name,
            thermal_field,
            pilot,
            navigator,
            aircraft,
            task)
        self._risk_managed = False

    def entry_actions(self):
        """
        """
        super(Optimize, self).entry_actions()
        plan = self.make_new_plan()
        self._pilot.set_plan(plan)
        self._pilot.navigate_plan(self._pilot.vehicle_state[:3])

    def make_new_plan(self, additional_exclude=[]):
        """Get a new destination

        Arguments:
            additional_exclude: optionally, thermals to exclude, a list

        Returns:
            new_destination: numpy (3,) array giving xyz position of destination
            is_risk_tolerance_met: whether risk is as low as desired
            plan: the plan we're gonna fly (thermals)
        """
        next_turnpoint = self._task.next_turnpoint

        location = self._pilot.vehicle_state[:3]
        wind = numpy.zeros((3,))

        airspeed = self._pilot.select_speed(0.0, 'optimize')
        sink_rate = self._aircraft.sink_rate(airspeed)
        glide_direction = next_turnpoint.X - location
        glide_direction /= numpy.linalg.norm(
            glide_direction * numpy.array([1.0, 1.0, 0.0]))

        ground_speed = numpy.linalg.norm(glide_direction * airspeed + wind)
        glide_range = ground_speed / sink_rate * self._pilot.vehicle_state[2]

        next_leg = self._task.next_leg
        if (
            self._task.glide_ratio_to_next_waypoint(location) < 10.0 and
            next_leg is not None):
            leg_distance = numpy.linalg.norm(
                (location - next_turnpoint.X) * numpy.array([1.0, 1.0, 0.0]))
            next_direction = numpy.diff(next_leg, axis=0)[0]
            next_direction /= numpy.linalg.norm(next_direction)
            next_leg = next_direction * (glide_range - leg_distance)
            flight_path = numpy.vstack((
                location,
                next_turnpoint.X,
                self._task.get_turnpoint(relative_idx=1).X))
            thermals = [
                self._task.next_turnpoint,
                self._task.get_turnpoint(relative_idx=1)]
        else:
            flight_path = numpy.vstack((
                location, location + glide_direction * glide_range))

            thermals = [self._task.next_turnpoint,]

        #TODO: this needs a better speed thing, probably send it MC so that it
        # properly speeds up for the upwind legs
        vision_range = -50.0 * location[2]
        glide_amoeba = self._pilot.glide_amoeba(airspeed, vision_range)
        reachable_thermals = self._thermal_field.reachable_thermals(
            glide_amoeba, additional_exclude)[0]

        # assign the closest thermal to our course
        P_landout = 1.0
        zero_3 = numpy.array([1.0, 1.0, 0.0])
        while (
            P_landout > self._pilot.risk_tolerance and
            len(reachable_thermals) > 0
            ):
            leg_lengths = numpy.linalg.norm(
                numpy.diff(flight_path, axis=0), axis=1)
            idx_to_expand = numpy.argmax(leg_lengths)
            leg_to_expand = flight_path[idx_to_expand:idx_to_expand + 2]
            path_to_expand = copy.deepcopy(flight_path) * zero_3

            ranges_to_course = numpy.fromiter(
                (numpy.linalg.norm(geometry.lines.point_line_distance(
                    thermal.X * zero_3, path_to_expand)[0]) for
                thermal in reachable_thermals), dtype=float)

            leg_lengths = numpy.linalg.norm(
                numpy.diff(flight_path, axis=0), axis=1)
            next_thermal = reachable_thermals.pop(
                numpy.argmin(ranges_to_course))
            if (
                next_thermal.id not in self._pilot.non_working_thermals and
                next_thermal.id not in self._pilot.visited_thermals):
                thermals.insert(idx_to_expand, next_thermal)
                flight_path = numpy.vstack((
                    flight_path[:idx_to_expand + 1],
                    thermals[idx_to_expand].X,
                    flight_path[idx_to_expand + 1:]))
                P_landout = self._pilot.check_plan(thermals, location)

            # Now we need to check if we can actually fly the whole plan
            leg_lengths = numpy.linalg.norm(
                numpy.diff(flight_path, axis=0), axis=1)
            cumulative_distance = numpy.cumsum(leg_lengths)
            iterator = itertools.izip(
                itertools.count(), cumulative_distance, thermals)
            for idx in range(len(thermals) - 2, 0, -1):
                if (
                    cumulative_distance[idx] > glide_range and not
                    isinstance(thermals[idx], task_module.Turnpoint)):
                    thermals.pop(idx)
                    flight_path = numpy.vstack((
                        flight_path[:idx+1], flight_path[idx+2:]))
                if cumulative_distance[idx] < glide_range:
                    break

            # sort our flight plan, start by finding where the turnpoints are
            sorted_idx = []
            break_indices = []
            for idx, wp in enumerate(thermals):
                if isinstance(wp, task_module.Turnpoint):
                    break_indices.append(idx+1)
            # now we're going to mark through the sections in between the
            # turnpoints and sort them by their distance to the turnpoint
            last_break = 0
            for break_idx in break_indices:
                distance_from_last = numpy.linalg.norm(
                    flight_path[last_break:break_idx] - flight_path[last_break],
                    axis=1)
                sorted_idx.extend(
                    (numpy.argsort(distance_from_last) + last_break).tolist())
                last_break = break_idx
            sorted_idx.append(break_idx)
            # now we need to create sorted versions of the flight path and
            # thermal list
            flight_path = flight_path[sorted_idx]
            thermals = [thermals[i-1] for i in sorted_idx[1:]]


        return thermals

    def _evaluate_risk(self, plan=None):
        """
        """
        risk = self._pilot.check_plan(plan)
        risk_acceptable = (
            risk <
            self._pilot._characteristics['P_landout_acceptable'])
        #print('risk: {}'.format(risk))
        return risk_acceptable

    def _assess_thermal_risk(self, thermal):
        """
        """
        possible_plan = self.make_new_plan([thermal.id,])
        risk = self._pilot.check_plan(possible_plan)
        print('skip_thermal_risk: {}'.format(risk))
        take_thermal = (
            risk > self._pilot._characteristics['P_landout_acceptable'])
        #take_thermal = not self._evaluate_risk(
        #    self.make_new_plan([thermal.id,]))
        return take_thermal

    def _check_transition_criteria(self):
        """
        """
        transition = super(Optimize, self)._check_transition_criteria()

        if transition:
            return transition

        if self._pilot.on_final_glide():
            return 'final_glide'

        if self._task.turnpoint_reached:
            return 'replan'

        found_thermal, thermal = self._pilot.detect_thermal()
        if found_thermal and thermal.id not in self._pilot.visited_thermals:
            if self._pilot.check_thermal(thermal) is not False:
                if self._assess_thermal_risk(thermal):
                    return 'thermal'
                else:
                    self._pilot.leaving_thermal(thermal.id, True)
                    return 'replan'
            elif self._pilot.thermal_in_plan(thermal):
                self._pilot.leaving_thermal(thermal.id, False)
                return 'replan'

        if not self._evaluate_risk():
            return 'minimize_risk'

        return None

class MinimizeRisk(SoaringState):
    """
    """
    def __init__(
        self,
        name,
        thermal_field,
        pilot,
        navigator,
        aircraft,
        task):
        """
        """
        super(MinimizeRisk, self).__init__(
            name,
            thermal_field,
            pilot,
            navigator,
            aircraft,
            task)

        self._risk_managed = False

    def entry_actions(self):
        """
        """
        super(MinimizeRisk, self).entry_actions()
        plan = self.make_new_plan()
        self._pilot.set_plan(plan)

    def recurrent_actions(self):
        self._pilot.navigate_plan(self._pilot.vehicle_state[:3])

    def make_new_plan(self):
        """Get a new destination

        Arguments:

        Returns:
            plan: the plan we're gonna fly (thermals)
        """
        next_turnpoint = self._task.next_turnpoint

        location = self._pilot.vehicle_state[:3]
        wind = numpy.zeros((3,))

        airspeed = self._pilot.select_speed(0.0, 'minimize_risk')
        sink_rate = self._aircraft.sink_rate(airspeed)
        glide_direction = next_turnpoint.X - location
        glide_direction /= numpy.linalg.norm(
            glide_direction * numpy.array([1.0, 1.0, 0.0]))

        glide_range = self._pilot.vehicle_state[2] / sink_rate * airspeed
        candidate_thermals = self._thermal_field.thermals_in_range(
            location, glide_range, self._pilot.non_working_thermals)[1]

        distance_to_turnpoint = numpy.linalg.norm(
            next_turnpoint.X - location * numpy.array([1.0, 1.0, 0.0]))

        candidate_thermal = None
        distance = numpy.inf
        for thermal in candidate_thermals:
            test_distance = thermal.distance(location)
            test_turnpoint_distance = thermal.distance(next_turnpoint.X)
            if (
                test_distance < distance and
                test_turnpoint_distance < distance_to_turnpoint and
                thermal.id not in self._pilot.visited_thermals):
                distance = test_distance
                candidate_thermal = thermal

        if candidate_thermal is None:
            return [next_turnpoint,]

        return [candidate_thermal,]

    def _check_transition_criteria(self):
        """
        """
        transition = super(MinimizeRisk, self)._check_transition_criteria()

        if transition:
            return transition

        if self._pilot.on_final_glide():
            return 'final_glide'

        if self._task.turnpoint_reached:
            return 'replan'

        found_thermal, thermal = self._pilot.detect_thermal()
        if found_thermal:
            if (
                self._pilot.check_thermal(thermal) is not False and
                thermal.id not in self._pilot.visited_thermals):
                if  self._pilot.vehicle_state[2] > -800:
                    return 'thermal'
                else:
                    self._pilot.leaving_thermal(thermal.id, True)
                    return 'optimize'
            elif self._pilot.thermal_in_plan(thermal):
                self._pilot.leaving_thermal(thermal.id, False)
                return 'replan'

        return None

class Survive(SoaringState):
    """
    """
    def __init__(
        self,
        name,
        thermal_field,
        pilot,
        navigator,
        aircraft,
        task):
        """
        """
        super(Survive, self).__init__(
            name,
            thermal_field,
            pilot,
            navigator,
            aircraft,
            task)

    def entry_actions(self):
        """
        """
        super(Survive, self).entry_actions()
        plan = self.make_new_plan()
        self._pilot.set_plan(plan)

    def recurrent_actions(self):
        self._pilot.navigate_plan(self._pilot.vehicle_state[:3])

    def make_new_plan(self):
        """Get a new destination

        Arguments:

        Returns:
            plan: the plan we're gonna fly (thermals)
        """
        next_turnpoint = self._task.next_turnpoint

        location = self._pilot.vehicle_state[:3]

        thermal = self._thermal_field.nearest(
            location, self._pilot.non_working_thermals)[0]

        return [thermal,]

    def _check_transition_criteria(self):
        """
        """
        transition = super(Survive, self)._check_transition_criteria()
        if transition:
            return transition

        if self._pilot.on_final_glide():
            return 'final_glide'

        if self._pilot.detect_thermal()[0]:
            return 'thermal'

        return None

class Thermal(SoaringState):
    """
    """
    def __init__(
        self,
        name,
        thermal_field,
        pilot,
        navigator,
        aircraft,
        task):
        """
        """
        super(Thermal, self).__init__(
            name,
            thermal_field,
            pilot,
            navigator,
            aircraft,
            task)
        self._thermal = None

    def entry_actions(self):
        """
        """
        super(Thermal, self).entry_actions()
        self._thermal = self._pilot.detect_thermal()[1]
        if self._thermal:
            self._pilot.thermal(self._thermal)

    def _working(self):
        """
        """
        if not self._thermal:
            return False
        return self._thermal.working(self._pilot.vehicle_state[:3])

    def exit_actions(self):
        """
        """
        super(Thermal, self).exit_actions()
        working = self._working()
        if working is not None and self._thermal:
            self._pilot.leaving_thermal(self._thermal.id, working)

    def _check_transition_criteria(self):
        """
        """
        transition = super(Thermal, self)._check_transition_criteria()

        if transition:
            return transition

        if self._pilot.on_final_glide():
            return 'final_glide'

        if (
            self._working() is False or
            -self._pilot.vehicle_state[2] > 0.98 * self._thermal.zmax):
            return 'optimize'

        return None

class FinalGlide(SoaringState):
    """
    """
    def __init__(
        self,
        name,
        thermal_field,
        pilot,
        navigator,
        aircraft,
        task):
        """
        """
        super(FinalGlide, self).__init__(
            name,
            thermal_field,
            pilot,
            navigator,
            aircraft,
            task)

    def entry_actions(self):
        """
        """
        super(FinalGlide, self).entry_actions()
        plan = self.make_new_plan()
        self._pilot.set_plan(plan)
        self._pilot.navigate_plan(self._pilot.vehicle_state[:3])

    def make_new_plan(self):
        """Get a new destination

        Arguments:

        Returns:
            new_destination: numpy (3,) array giving xyz position of destination
            is_risk_tolerance_met: whether risk is as low as desired
            plan: the plan we're gonna fly (thermals)
        """
        next_turnpoint = self._task.next_turnpoint
        return [next_turnpoint,]

    def _check_transition_criteria(self):
        """
        """
        transition = super(FinalGlide, self)._check_transition_criteria()

        if transition:
            return transition

        if self._task.turnpoint_reached:
            return 'replan'

        if not self._pilot.on_final_glide():
            return 'below_glide'

        return None
