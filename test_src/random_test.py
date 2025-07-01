import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt

# number of signal points
N = 400
# sample spacing
T = 1.0 / 80.0
x = np.linspace(0.0, N * T, N, endpoint=False)

w = 10
dx = 10 * np.pi * 1e-4
size = np.size(x)
x_fft = fft.fftfreq(N, T)
x_fft = fft.fftshift(x_fft)
y = np.cos(w * x)
y_fft = fft.fft(y)
y_fft = fft.fftshift(y_fft)
plt.plot(x_fft, abs(y_fft))
plt.show(block=True)
