import pdb

import numpy

import matplotlib.pyplot as plt

class StateRecord(object):
    """Holds a record of a state
    """
    def __init__(self, state_dim=None, input_dim=None):
        """Constructor

        Arguments
            state_dim: tuple, the dimensions of the state record to hold. If
                left unspecified then the record will be stored as an expanding
                list. If this dimension is overrun then the remainder will be
                stored as an expanding list. The first entry is the number of
                entries.
            input_dim: tuple, the dimensions of the input record to hold. If
                left unspecified then the record will be stored as an expanding
                list. If this dimension is overrun then the remainder will be
                stored as an expanding list. The first entry is the number of
                entries. The number of entries in state and input must match

        Returns:
            class instance
        """
        if isinstance(state_dim, tuple) and isinstance(input_dim, tuple):
            assert state_dim[0] == input_dim[0],\
                'state and input entry sizes inconsistent'
            self._state_dim = state_dim
            self._input_dim = input_dim
            self._state_record = numpy.zeros(state_dim)
            self._input_record = numpy.zeros(input_dim)
            self._time = numpy.zeros((state_dim[0],))
            self._idx = 1
        else:
            self._state_dim = None
            self._input_dim = None
            self._time = []
            self._state_record = []
            self._input_record = []

        self._time_padding = []
        self._state_padding = []
        self._input_padding = []

    def reset(self, t0, X0):
        """reset the record

        Arguments:
            t0: initial state
            X0: initial state

        Returns:
            no returns
        """
        self._time_padding = []
        self._state_padding = []
        self._input_padding = []

        if self._state_dim is None:
            self._time = [t0,]
            self._state_record = [X0,]
            self._input_record = []

        self._state_record = numpy.zeros(self._state_dim)
        self._input_record = numpy.zeros(self._input_dim)
        self._time = numpy.zeros((self._state_dim[0],))

        self._state_record[0] = X0
        self._time[0] = t0

        self._idx = 1

    def update_state(self, t, X, U):
        """Update the state

        Arguments:
            t: the time stamp
            X: the new state
            U the new input

        Returns:
            no returns
        """
        if self._state_dim is None:
            self._time.append(t)
            self._state_record.append(X)
            self._input_record.append(U)
            self._idx += 1
            return

        if self._idx > self._state_dim[0]:
            self._time_padding.append(t)
            self._state_padding.append(X)
            self._input_padding.append(U)
            self._idx += 1
            return

        self._time[self._idx] = t
        self._state_record[self._idx] = X
        self._input_record[self._idx - 1] = U
        self._idx += 1

    def plot(self, state_idx=[], input_idx=[], plot_slice=None, axes=None):
        """Plot a record

        Arguments:
            state_idx: tuple of the indices of the state records to plot
            input_idx: tuple of the indices of the input records to plot
            plot_slice: indices of the time record to plot
            axes: axes to plot into.

        Returns:
            no returns
        """
        if axes is None:
            # haxx0r stuff to let me use the same code. this graps the handle
            # to the matplotlib library...
            axes = plt

        if plot_slice is None:
            plot_slice = range(self._idx)

        for idx in state_idx:
            axes.plot(self.t[plot_slice], self.X[plot_slice, idx])

        for idx in input_idx:
            axes.plot(self.t[plot_slice], self.U[plot_slice, idx])

    @property
    def t(self):
        """Getter for a unified time as a numpy array
        """
        if self._state_dim is None:
            return numpy.array(self._time)
        elif self._idx < self._state_dim[0]:
            return self._time[:self._idx]
        return numpy.hstack(self._time, self._time_padding)

    @property
    def X(self):
        """Getter for unified state history as numpy array
        """
        if self._state_dim is None:
            return numpy.array(self._state_record)
        elif self._idx < self._state_dim[0]:
            return self._state_record[:self._idx]
        return numpy.hstack(self._state_record, self._state_padding)

    @property
    def U(self):
        """Getter for unified input history as numpy array
        """
        if self._input_dim is None:
            return numpy.array(self._input_record)
        elif self._idx < self._state_dim[0]:
            return self._input_record[:self._idx]
        return numpy.hstack(self._input_record, self._input_padding)
