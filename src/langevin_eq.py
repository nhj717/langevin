import numpy as np
import scipy.constants as const
import scipy.fft as fft
import matplotlib.pyplot as plt
import beam_profile


def gamma(radius, density, cross_section, eta, pressure, T):
    mass = 4 * np.pi / 3 * radius**3 * density
    l_fp = const.k * T / (np.sqrt(2) * cross_section * pressure)
    Kn = l_fp / radius
    ck = 1 / (1 + Kn * (1.231 + 0.469 * np.exp(-1.178 / Kn)))
    gamma = 6 * np.pi * eta * radius * ck / mass
    return gamma


class Langevin:
    def __init__(self):
        # units
        mW = 1e-3
        um = 1e-6
        nm = 1e-9
        mbar = 100

        # physical values
        self.wl = 1.064 * um
        self.k = 2 * np.pi / self.wl
        self.T = 273  # kelvins
        radius = 300 / 2 * nm
        density = 2.2e3  # kgm-3
        cross_section = (
            np.pi * (0.36 * nm) ** 2
        )  # mean cross-section of the air molecules
        pressure = 0.5 * mbar
        eta = 2.791 * 1e-7 * self.T**0.7355  # viscosity coefficient of the air    m^2/s
        self.m = density * 4 / 3 * np.pi * radius**3
        self.gamma0 = gamma(radius, density, cross_section, eta, pressure, self.T)
        self.P = 500 * mW
        self.r_core = 20 * um
        eps_glass = 3.9
        alpha0 = (
            4 * np.pi * const.epsilon_0 * radius**3 * (eps_glass - 1) / (eps_glass + 2)
        )
        self.alpha = alpha0 / (
            1 - 1j * alpha0 * self.k**3 / (6 * np.pi * const.epsilon_0)
        )

        self.t_f = 1e-1  # final time in sec
        self.delt = 1e-6  # resoltion of the time array
        self.t = np.arange(0, self.t_f, self.delt)
        self.array_size = np.size(self.t)
        self.f = np.linspace(0, 1 / self.delt, self.array_size)

    def langevin_eq(self):
        x0 = np.zeros(2)  # initial position in m
        v0 = np.zeros(2)  # initial velocity in m/s

        # Optical force
        f_opt_r, f_opt_phi = beam_profile.gaussian_standing_wave(
            self.P, self.r_core, self.alpha
        )

        # Thermal force
        noise = np.random.randn(self.array_size)  # noise
        print(np.mean(noise))
        plt.hist(noise, bins=100)
        plt.show(block=True)
        f_therm = np.sqrt(2 * const.k * self.T * self.gamma0 / self.m) * noise

        x = np.zeros((2, self.array_size))
        v = np.zeros_like(x)
        x[:, 0] = x0
        v[:, 0] = v0

        for i in range(self.array_size - 1):
            theta = np.arctan2(x[1, i], x[0, i])
            f_opt = np.array([np.cos(theta), np.sin(theta)]) * f_opt_r(x[0, i], x[1, i])
            v[:, i + 1] = v[:, i] + self.delt * (
                -self.gamma0 * v[:, i] + f_opt / self.m + f_therm[i]
            )
            x[:, i + 1] = x[:, i] + v[:, i + 1] * self.delt
        self.x = x
        self.v = v

    def plot(self):
        plt.plot(self.t, self.x[0, :])
        plt.xlabel("Time [s]")
        plt.ylabel("X [m]")
        plt.show(block=True)
        plt.plot(self.x[0, :], self.m * self.v[0, :])
        plt.xlabel("X [m]")
        plt.ylabel("P [kg*m/s]")
        plt.show(block=True)

        x_fft = fft.fft(self.x[0, :])
        x_fft = fft.fftshift(x_fft)
        plt.plot(self.f, x_fft * 1e-3)
        plt.xlabel("f [kHz]")
        plt.ylabel("S [a.u.]")
        plt.show(block=True)
