# Functions for optical forces in cylindrical coordinate
import scipy.special as sp
import numpy as np
import scipy.constants as const


def f_gaussian_standing_wave(P, r_core, alpha, beta):
    u01 = sp.jn_zeros(0, 1)
    factor = (
        2
        * P
        * np.real(alpha)
        / (np.pi * r_core**2 * const.c * const.epsilon_0 * sp.jve(1, u01) ** 2)
    )
    f_r = (
        lambda x, y, z: -2
        * factor
        * sp.jve(0, u01 * np.sqrt(x**2 + y**2) / r_core)
        * sp.jve(1, u01 * np.sqrt(x**2 + y**2) / r_core)
        * u01
        / r_core
        * np.cos(beta * z) ** 2
    )
    f_phi = lambda x, y, z: 0
    f_z = (
        lambda x, y, z: -factor
        * sp.jve(0, u01 * np.sqrt(x**2 + y**2) / r_core) ** 2
        * np.sin(2 * beta * z)
        * beta
    )
    return f_r, f_phi, f_z


def f_oam_standing_wave(P, r_core, alpha, beta, l):
    ul1 = sp.jn_zeros(l, 1)
    norm = 1 / (-np.pi * sp.jve(l - 1, ul1) * sp.jve(l + 1, ul1) * r_core**2)
    factor = 2 * P * norm / const.c / const.epsilon_0
    f_r = (
        lambda x, y, z: factor
        * np.real(alpha)
        * sp.jve(l, ul1 * np.sqrt(x**2 + y**2) / r_core)
        * (
            sp.jve(l - 1, ul1 * np.sqrt(x**2 + y**2) / r_core)
            - sp.jve(l + 1, ul1 * np.sqrt(x**2 + y**2) / r_core)
        )
        * ul1
        / r_core
        * np.cos(beta * z) ** 2
    )
    f_phi = (
        lambda x, y, z: factor
        * l
        / np.sqrt(x**2 + y**2)
        * np.imag(alpha)
        * sp.jve(l, ul1 * np.sqrt(x**2 + y**2) / r_core) ** 2
        * np.cos(beta * z) ** 2
    )
    f_z = (
        lambda x, y, z: -factor
        * np.real(alpha)
        * sp.jve(l, ul1 * np.sqrt(x**2 + y**2) / r_core) ** 2
        * np.sin(2 * beta * z)
        * beta
    )
    return f_r, f_phi, f_z
