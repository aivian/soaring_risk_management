import numpy

import scipy

class SailplanePilot(object):
    """A decision-making class
    """
    def __init__(self, thermal_field):
        """Constructor

        Arguments:
            thermal_field: the thermal environment this pilot is flying in

        Returns:
            class instance
        """

