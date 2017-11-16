import numpy

import thermal_field

import matplotlib.pyplot as plt

t = thermal_field.ThermalField(100000.0, 1000.0, 0.0, 500)

x = [therm._x for therm in t._thermals]
x = numpy.array(x)

plt.scatter(x[:,0], x[:,1])
plt.grid()
plt.show()
