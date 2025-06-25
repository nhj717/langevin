import numpy as np

x = [1,1j]
y = x/np.linalg.norm(x)
print(y)