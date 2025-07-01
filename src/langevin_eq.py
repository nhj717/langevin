import numpy as np
import scipy.constants as const
import scipy.fft as fft
from scipy.special import jn_zeros
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import beam_profile


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
        self.T = 300  # kelvins
        radius = 300 / 2 * nm
        density = 2.2e3  # kgm-3
        cross_section = (
            np.pi * (0.36 * nm) ** 2
        )  # mean cross-section of the air molecules
        pressure = 10 * mbar
        eta = 2.791 * 1e-7 * self.T**0.7355  # viscosity coefficient of the air    m^2/s
        self.m = density * 4 / 3 * np.pi * radius**3
        self.gamma0 = gamma(radius, density, cross_section, eta, pressure, self.T)
        self.P = 500 * mW  # power on each side
        self.r_core = 20 * um
        self.beta = (
            2
            * np.pi
            / self.wl
            * (1 - 1 / 2 * (jn_zeros(0, 1) * self.wl / 2 / np.pi / self.r_core) ** 2)
        )
        eps_glass = 3.9
        alpha0 = (
            4 * np.pi * const.epsilon_0 * radius**3 * (eps_glass - 1) / (eps_glass + 2)
        )
        self.alpha = alpha0 / (
            1 - 1j * alpha0 * self.k**3 / (6 * np.pi * const.epsilon_0)
        )

        t_f = 1e-2  # final time in sec
        # Number of sample points
        self.N = int(1e5)
        # sample spacing
        self.delt = 1e-7  # resolution of the time array
        self.t = np.linspace(0, self.N * self.delt, self.N)
        self.f = fft.fftfreq(self.N, self.delt)[: int(self.N / 2)]
        self.omega = 2 * np.pi * self.f

    def langevin_eq(self):
        x0 = np.zeros(3)  # initial position in m
        v0 = np.zeros(3)  # initial velocity in m/s

        # Optical force
        f_opt_r, f_opt_phi, f_opt_z = beam_profile.gaussian_standing_wave(
            self.P, self.r_core, self.alpha, self.beta
        )
        # Thermal force
        noise = np.random.randn((3,self.N))  # noise
        f_therm = np.sqrt(2 * const.k * self.T * self.gamma0 / self.m) * noise

        x = np.zeros((3, self.N))
        v = np.zeros_like(x)
        x[:, 0] = [1e-10, 1e-10, 1e-11]
        v[:, 0] = v0

        for i in range(self.N - 1):
            theta = np.arctan2(x[1, i], x[0, i])
            f_opt = np.array(
                [np.cos(theta), np.sin(theta), 0] * f_opt_r(x[0, i], x[1, i], x[2, i])
                + [0, 0, 1] * f_opt_z(x[0, i], x[1, i], x[2, i])
            )
            v[:, i + 1] = v[:, i] + self.delt * (
                -self.gamma0 * v[:, i] + f_opt / self.m + f_therm[:,i]
            )
            x[:, i + 1] = x[:, i] + v[:, i + 1] * self.delt
            check = f_opt / self.m
        self.x = x
        self.v = v

    def plot_x(self):
        plt.plot(self.t, self.x[0, :])
        plt.xlabel("Time [s]")
        plt.ylabel("X [m]")
        plt.show(block=True)
        plt.plot(self.x[0, :], self.m * self.v[0, :])
        plt.xlabel("X [m]")
        plt.ylabel("P [kg*m/s]")
        plt.show(block=True)

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
        print(
            f"Peak position is {lorentzian_fit_coeff[0]} rad. Hz and the amplitude is {lorentzian_fit_coeff[1]}"
        )
        print(
            f"Actual gamma0 is {self.gamma0 / (2 * np.pi)}Hz and the calculated gamma0 is {lorentzian_fit_coeff[2] / (2 * np.pi)}Hz"
        )
        plt.plot(self.f * 1e-3, np.log10(x_fft))
        plt.plot(self.f * 1e-3, np.log10(x_fft_fit))
        # plt.xlim(0.1, 10000)
        plt.xlabel("f [kHz]")
        plt.ylabel("S [a.u.]")
        plt.show(block=True)

    def plot_z(self):
        plt.plot(self.t, self.x[2, :])
        plt.xlabel("Time [s]")
        plt.ylabel("Z [m]")
        plt.show(block=True)
        plt.plot(self.x[2, :], self.m * self.v[2, :])
        plt.xlabel("Z [m]")
        plt.ylabel("P [kg*m/s]")
        plt.show(block=True)

        x_fft = 2.0 / self.N * fft.fft(self.x[2, :])[: int(self.N / 2)]
        x_fft = abs(x_fft) ** 2
        peak_w_z = self.omega[np.argmax(x_fft)]
        lorentzian_fit_coeff, lorentzian_fit_error2 = curve_fit(
            lorentzian, self.omega, x_fft, p0=[peak_w_z, 5e-6, self.gamma0]
        )
        x_fft_fit = lorentzian(
            self.omega,
            lorentzian_fit_coeff[0],
            lorentzian_fit_coeff[1],
            lorentzian_fit_coeff[2],
        )
        print(
            f"Peak position is {lorentzian_fit_coeff[0]} rad. Hz and the amplitude is {lorentzian_fit_coeff[1]}"
        )
        print(
            f"Actual gamma0 is {self.gamma0 / (2 * np.pi) }Hz and the calculated gamma0 is {lorentzian_fit_coeff[2] / (2 * np.pi) }Hz"
        )
        plt.plot(self.f * 1e-3, np.log10(x_fft))
        plt.plot(self.f * 1e-3, np.log10(x_fft_fit))
        # plt.xlim(0.1, 10000)
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
            f"Actual gamma0 is {self.gamma0 / (2 * np.pi)}Hz and the calculated gamma0 is {lorentzian_fit_coeff[2] / (2 * np.pi)}Hz for x and {lorentzian_fit_coeff2[2] / (2 * np.pi)}Hz for z"
        )

        plt.plot(self.f * 1e-3, np.log10(x_fft), "orange", label="xfft")
        plt.plot(self.f * 1e-3, np.log10(x_fft_fit), "red", label="xfft fit")
        plt.plot(self.f * 1e-3, np.log10(z_fft), "green", label="zfft")
        plt.plot(self.f * 1e-3, np.log10(z_fft_fit), "blue", label="zfft fit")
        plt.xlim(0.1, 200)
        plt.xlabel("f [kHz]")
        plt.ylabel("S [a.u.]")
        plt.legend()
        plt.show(block=True)
