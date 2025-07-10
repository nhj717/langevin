import numpy as np
import scipy.constants as const
import scipy.fft as fft
from scipy.special import jn_zeros
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import beam_profile


def initial_setup():
    diameter = 400  # in nanometers
    eps_glass = 3.9
    power = 100  # in mW from both sides
    pressure = 1  # in mbar
    core_radius = 22  # in um
    N = int(1e5)  # Total number of sampling
    delt = 1e-6  # in seconds, time resolution of the simulation
    iteration = 10  # number of sampling
    return diameter, eps_glass, power, pressure, core_radius, N, delt, iteration


def gamma(radius, density, cross_section, eta, pressure, T):
    mass = 4 * np.pi / 3 * radius**3 * density
    l_fp = const.k * T / (np.sqrt(2) * cross_section * pressure)
    Kn = l_fp / radius
    ck = 1 / (1 + Kn * (1.231 + 0.469 * np.exp(-1.178 / Kn)))
    gamma = 6 * np.pi * eta * radius * ck / mass
    return gamma


def lorentzian(x, x0, a, gamma):
    x0 = 2 * np.pi * x0
    x = 2 * np.pi * x
    gamma = 2 * np.pi * gamma
    return a * gamma / ((x0**2 - x**2) ** 2 + (x * gamma) ** 2)


def coord_trafo(A, theta):
    B = np.zeros_like(A, dtype="complex128")
    B[0, :] = A[0, :] * np.cos(theta) + A[1, :] * np.sin(theta)
    B[1, :] = -A[0, :] * np.sin(theta) + A[1, :] * np.cos(theta)
    B[2, :] = A[2, :]
    return B


class oam_Langevin:
    def __init__(
        self, diameter, eps_glass, power, pressure, core_radius, N, delt, iteration
    ):
        # units
        mW = 1e-3
        um = 1e-6
        nm = 1e-9
        mbar = 100

        # physical values
        self.wl = 1.064 * um
        self.k = 2 * np.pi / self.wl
        self.T = 300  # kelvins
        radius = diameter / 2 * nm
        density = 2.2e3  # kgm-3
        cross_section = (
            np.pi * (0.36 * nm) ** 2
        )  # mean cross-section of the air molecules
        pressure = pressure * mbar
        eta = 2.791 * 1e-7 * self.T**0.7355  # viscosity coefficient of the air    m^2/s
        self.m = density * 4 / 3 * np.pi * radius**3
        self.gamma0 = gamma(radius, density, cross_section, eta, pressure, self.T)
        self.P = power * mW  # power on each side
        self.r_core = core_radius * um
        self.beta = (
            2
            * np.pi
            / self.wl
            * (1 - 1 / 2 * (jn_zeros(0, 1) * self.wl / 2 / np.pi / self.r_core) ** 2)
        )
        alpha0 = (
            4 * np.pi * const.epsilon_0 * radius**3 * (eps_glass - 1) / (eps_glass + 2)
        )
        self.alpha = alpha0 / (
            1 - 1j * alpha0 * self.k**3 / (6 * np.pi * const.epsilon_0)
        )

        self.iteration = iteration  # number of iterations
        self.N = N  # Number of sample points
        self.delt = delt  # resolution of the time array
        self.t = np.linspace(0, self.N * self.delt, self.N)
        self.f = fft.fftfreq(self.N, self.delt)[: int(self.N / 2)]
        self.f_start = int(np.abs(self.f - 50).argmin())
        self.omega = 2 * np.pi * self.f
        self.x = np.zeros((3, self.N))
        self.v = np.zeros_like(self.x)

    def langevin_eq(self):

        # Optical force
        f_opt_r, f_opt_phi, f_opt_z = beam_profile.gaussian_standing_wave(
            self.P, self.r_core, self.alpha, self.beta
        )
        # Thermal force
        factor = np.sqrt(2 * const.k * self.T * self.m * self.gamma0)
        f_therm = factor * np.random.randn(self.iteration, 3, self.N)
        gravity = np.array([0, 1, 0]) * np.ones(self.iteration)[:, None] * (-9.8)

        x = np.zeros((self.iteration, 3, 2))
        v = np.zeros_like(x)
        # x[:, :, 0] = [1e-10, 1e-10, 1e-11]
        # v[:, :, 0] = 0
        x[:, :, 0] = np.random.randn(self.iteration, 3) * 1e-11
        v[:, :, 0] = np.random.randn(self.iteration, 3) * 1e-5
        self.x[:, 0] = np.average(x[:, :, 0], axis=0)
        self.v[:, 0] = np.average(v[:, :, 0], axis=0)

        for i in range(self.N - 1):
            theta = np.arctan2(x[:, 1, 0], x[:, 0, 0])
            f_opt = (
                np.array(
                    [np.cos(theta), np.sin(theta), np.zeros_like(theta)]
                    * f_opt_r(x[:, 0, 0], x[:, 1, 0], x[:, 2, 0])
                ).T
                + np.array([0, 0, 1])
                * f_opt_z(x[:, 0, 0], x[:, 1, 0], x[:, 2, 0])[:, None]
            )

            v[:, :, 1] = v[:, :, 0] + self.delt * (
                -self.gamma0 * v[:, :, 0]
                + f_opt / self.m
                + f_therm[:, :, i] / self.m
                + gravity
            )
            x[:, :, 1] = x[:, :, 0] + v[:, :, 1] * self.delt

            self.x[:, i + 1] = np.average(x[:, :, 1], axis=0)
            self.v[:, i + 1] = np.average(v[:, :, 1], axis=0)

            x[:, :, 0] = x[:, :, 1]
            v[:, :, 0] = v[:, :, 1]

    def plot(self, xyz):
        if xyz == "x":
            index = 0
        elif xyz == "y":
            index = 1
        else:
            index = 2
        plt.plot(self.t, self.x[index, :])
        plt.xlabel("Time [s]")
        plt.ylabel(f"{xyz} [m]")
        plt.show(block=True)
        plt.plot(self.x[index, :], self.m * self.v[index, :])
        plt.xlabel(f"{xyz} [m]")
        plt.ylabel("P [kg*m/s]")
        plt.show(block=True)

        x_fft = 2.0 / self.N * fft.fft(self.x[index, :])[: int(self.N / 2)]
        self.x_fft = abs(x_fft) ** 2
        plt.plot(self.f * 1e-3, np.log10(x_fft))
        plt.xlim(0.5, 100)
        plt.xlabel("f [kHz]")
        plt.ylabel("S [a.u.]")
        plt.show(block=True)

    def plot_spectrums(self):
        x_fft = 2.0 / self.N * fft.fft(self.x[0, :])[: int(self.N / 2)]
        x_fft = abs(x_fft) ** 2
        peak_w_x = self.omega[np.argmax(x_fft)]
        lorentzian_fit_coeff, lorentzian_fit_error = curve_fit(
            lorentzian, self.omega, x_fft, p0=[peak_w_x, 5e-6, self.gamma0]
        )
        x_fft_fit = lorentzian(
            self.omega,
            lorentzian_fit_coeff[0],
            lorentzian_fit_coeff[1],
            lorentzian_fit_coeff[2],
        )
        y_fft = 2.0 / self.N * fft.fft(self.x[1, :])[: int(self.N / 2)]
        y_fft = abs(y_fft) ** 2
        peak_w_y = self.omega[np.argmax(y_fft[self.f_start :])]
        lorentzian_fit_coeff1, lorentzian_fit_error1 = curve_fit(
            lorentzian,
            self.omega[self.f_start :],
            y_fft[self.f_start :],
            p0=[peak_w_y, 5e-6, self.gamma0],
        )
        y_fft_fit = lorentzian(
            self.omega,
            lorentzian_fit_coeff1[0],
            lorentzian_fit_coeff1[1],
            lorentzian_fit_coeff1[2],
        )
        z_fft = 2.0 / self.N * fft.fft(self.x[2, :])[: int(self.N / 2)]
        z_fft = abs(z_fft) ** 2
        peak_w_z = self.omega[np.argmax(z_fft)]
        lorentzian_fit_coeff2, lorentzian_fit_error2 = curve_fit(
            lorentzian, self.omega, z_fft, p0=[peak_w_z, 5e-6, self.gamma0]
        )
        z_fft_fit = lorentzian(
            self.omega,
            lorentzian_fit_coeff2[0],
            lorentzian_fit_coeff2[1],
            lorentzian_fit_coeff2[2],
        )

        print(
            f"Actual gamma0 is {self.gamma0 / (2 * np.pi)}Hz and the calculated gamma0 is {lorentzian_fit_coeff[2] / (2 * np.pi)}Hz for x, {lorentzian_fit_coeff1[2] / (2 * np.pi)}Hz for y and {lorentzian_fit_coeff2[2] / (2 * np.pi)}Hz for z"
        )

        plt.plot(self.f * 1e-3, np.log10(x_fft), "orange", label="xfft")
        plt.plot(self.f * 1e-3, np.log10(x_fft_fit), "red", label="xfft fit")
        plt.plot(self.f * 1e-3, np.log10(y_fft), "cyan", label="yfft")
        plt.plot(self.f * 1e-3, np.log10(y_fft_fit), "blue", label="yfft fit")
        plt.plot(self.f * 1e-3, np.log10(z_fft), "brown", label="zfft")
        plt.plot(self.f * 1e-3, np.log10(z_fft_fit), "black", label="zfft fit")
        plt.xlim(0.1, 100)
        plt.xlabel("f [kHz]")
        plt.ylabel("S [a.u.]")
        plt.legend()
        plt.show(block=True)
