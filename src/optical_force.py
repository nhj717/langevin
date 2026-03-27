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


def E_field(direction=1):
    """
    Transverse Electric field calculated in 3D cart. coord.
    """
    if direction == 1:
        l = self.l1
        u_lm = self.u_lm
    elif direction == -1:
        l = self.l2
        u_lm = self.u_lm_count

    beta = direction * self.beta
    xxx, yyy, zzz = np.meshgrid(self.x, self.y, self.z, indexing="ij")
    rrr = np.sqrt(xxx**2 + yyy**2)
    ppphi = np.arctan2(yyy, xxx)
    self.ppphi = ppphi
    Ex_cart = np.zeros((3, len(self.x), len(self.y), len(self.z)), dtype="complex128")
    Ey_cart = np.zeros_like(Ex_cart, dtype="complex128")
    Bx_cart = np.zeros_like(Ex_cart, dtype="complex128")
    By_cart = np.zeros_like(Ex_cart, dtype="complex128")

    Ex_cart[0, :] = self.u(u_lm, l, rrr, ppphi)
    Ex_cart[1, :] = 0
    Ex_cart[2, :] = 1j / beta * self.dxu(u_lm, l, rrr, ppphi)
    Ex_cart = (1j) * self.w0 * Ex_cart * np.exp(1j * beta * zzz)

    Ey_cart[0, :] = 0
    Ey_cart[1, :] = self.u(u_lm, l, rrr, ppphi)
    Ey_cart[2, :] = 1j / beta * self.dyu(u_lm, l, rrr, ppphi)
    Ey_cart = (1j) * self.w0 * Ey_cart * np.exp(1j * beta * zzz)

    return Ex_cart, Ey_cart


def f_opt_general(P, r_core, alpha, beta, l):
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
