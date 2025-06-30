import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import beam_profile

t_f = 1e-3  # final time in sec
delt = 1e-6  # resoltion of the time array
t = np.arange(0, t_f, delt)

array_size = np.size(t)
x0 = np.zeros(2)  # initial position in m
v0 = np.zeros(2)  # initial velocity in m/s
a0 = np.zeros(2)  # initial acceleration in m/s2


m = 1  # mass in kg
gamma = 0.7  # damping coefficient/mass in Hz

# Optical force
f_opt = beam_profile.gaussian_standing_wave()

#Thermal force
noise = np.random.randn(array_size)       #noise
print(np.mean(noise))
plt.hist(noise,bins = 100)
plt.show(block = True)
f_therm = np.sqrt(2*kb*T*gamma/m)

x = np.zeros(2, array_size)
v = np.zeros(2, array_size)
x[:, 0] = x0
v[:, 0] = v0

for i in range(array_size - 1):
    x[:, i + 1] = x[:, i] + v[:, i] * delt
    v[:, i + 1] = v[:, i] + delt * (-gamma * v[:, i] + f_opt(x[0,i+1],x[1,i+1]))+f_therm

plt.plot(t, x)
plt.xlabel("Time [s]")
plt.ylabel("X [m]")
plt.show(blocked=True)
plt.plot(x, m * v)
plt.xlabel("X [m]")
plt.ylabel("P [kg*m/s]")
plt.show(blocked=True)
