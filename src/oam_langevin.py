import numpy as np
import scipy.constants as const
from scipy.special import jn_zeros
import scipy.fft as fft
from optical_force import f_oam_standing_wave
import h5py


def gamma(radius, density, cross_section, eta, pressure, T):
    mass = 4 * np.pi / 3 * radius**3 * density
    l_fp = const.k * T / (np.sqrt(2) * cross_section * pressure)
    Kn = l_fp / radius
    ck = 1 / (1 + Kn * (1.231 + 0.469 * np.exp(-1.178 / Kn)))
    gamma = 6 * np.pi * eta * radius * ck / mass
    return gamma


class oam_Langevin:
    def __init__(
        self,
        diameter,
        eps_glass,
        power,
        pressure,
        core_radius,
        N,
        delt,
        iteration,
        mode_number,
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
            * (
                1
                - 1
                / 2
                * (jn_zeros(mode_number, 1) * self.wl / 2 / np.pi / self.r_core) ** 2
            )
        )  # propagation constant of the fiber mode
        alpha0 = (
            4 * np.pi * const.epsilon_0 * radius**3 * (eps_glass - 1) / (eps_glass + 2)
        )
        self.alpha = alpha0 / (
            1 - 1j * alpha0 * self.k**3 / (6 * np.pi * const.epsilon_0)
        )  # susceptibility from the dipole approximation

        self.iteration = iteration
        self.N = N
        self.delt = delt
        self.t = np.linspace(0, self.N * self.delt, self.N)
        self.f = fft.fftfreq(self.N, self.delt)[: int(self.N / 2)]
        self.f_start = int(np.abs(self.f - 50).argmin())
        self.omega = 2 * np.pi * self.f
        self.x = np.zeros((3, self.N))
        self.v = np.zeros_like(self.x)
        self.mode_number = mode_number

    def langevin_eq(self):
        # Optical force
        f_opt_r, f_opt_phi, f_opt_z = f_oam_standing_wave(
            self.P, self.r_core, self.alpha, self.beta, self.mode_number
        )
        # Thermal force
        factor = np.sqrt(2 * const.k * self.T * self.gamma0 / self.m)
        f_therm = factor * np.random.randn(self.iteration, 3, self.N)

        # gravity
        gravity = np.array([0, 1, 0]) * np.ones(self.iteration)[:, None] * (-9.8)

        x = np.zeros((self.iteration, 3, 2))
        v = np.zeros_like(x)

        # set initial conditions for x and v
        x[:, :, 0] = np.random.randn(self.iteration, 3) * 1e-11
        if self.mode_number == 0:
            x[:, 1, 0] -= 1e-6
        else:
            x[:, 1, 0] += 1e-5
        v[:, :, 0] = 0
        self.x[:, 0] = np.average(x[:, :, 0], axis=0)
        self.v[:, 0] = np.average(v[:, :, 0], axis=0)

        for i in range(self.N - 1):
            # find theta from the given position
            theta = np.arctan2(x[:, 1, 0], x[:, 0, 0])

            # Calculate for the total optical force in cartesian coordinate
            f_opt = (
                np.array(
                    [np.cos(theta), np.sin(theta), np.zeros_like(theta)]
                    * f_opt_r(x[:, 0, 0], x[:, 1, 0], x[:, 2, 0])
                ).T
                + np.array(
                    [-np.sin(theta), np.cos(theta), np.zeros_like(theta)]
                    * f_opt_phi(x[:, 0, 0], x[:, 1, 0], x[:, 2, 0])
                ).T
                + np.array([0, 0, 1])
                * f_opt_z(x[:, 0, 0], x[:, 1, 0], x[:, 2, 0])[:, None]
            )

            # Euler's method to update for the velocity from the given force
            v[:, :, 1] = v[:, :, 0] + self.delt * (
                -self.gamma0 * v[:, :, 0] + f_opt / self.m + f_therm[:, :, i] + gravity
            )

            # Euler's method to update for the position from the given velocity
            x[:, :, 1] = x[:, :, 0] + v[:, :, 1] * self.delt

            # to save memory, current step is averaged over the iteration and saved to self.x and self.v
            self.x[:, i + 1] = np.average(x[:, :, 1], axis=0)
            self.v[:, i + 1] = np.average(v[:, :, 1], axis=0)

            # also within the loop, only the previous step is remembered(Markovian system)
            x[:, :, 0] = x[:, :, 1]
            v[:, :, 0] = v[:, :, 1]

    def save_data(self, location, file_name, group_name):
        with h5py.File("{}/{}.h5".format(location, file_name), "a") as hdf:
            try:
                group = hdf.create_group(group_name)
                group.create_dataset("m", data=self.m)
                group.create_dataset("N", data=self.N)
                group.create_dataset("gamma0", data=self.gamma0)
                group.create_dataset("t", data=self.t)
                group.create_dataset("f", data=self.f)
                group.create_dataset("x", data=self.x)
                group.create_dataset("v", data=self.v)
            except:
                del hdf[group_name]
                group = hdf.create_group(group_name)
                group.create_dataset("m", data=self.m)
                group.create_dataset("N", data=self.N)
                group.create_dataset("gamma0", data=self.gamma0)
                group.create_dataset("t", data=self.t)
                group.create_dataset("f", data=self.f)
                group.create_dataset("x", data=self.x)
                group.create_dataset("v", data=self.v)
            hdf.close()
        print("success")
