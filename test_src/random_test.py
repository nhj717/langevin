import numpy as np

x = np.array([0,0,1])
y = np.array([2,3])
z = x[:,None]*y
c = x[:,None]
print(np.shape(c))
print(c)
