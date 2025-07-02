import numpy as np

x = np.array([0, 1])
x[0] = x[1]
x[1] = 2
print(np.shape(x))
print(x)
