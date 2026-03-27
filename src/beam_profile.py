"""
Functions in this folder are used to visualize the beam profiles of LP modes.
Direction of the Poynting vector was the main interest in the OAM profile class.
"""

import h5py
import numpy as np
from scipy import constants as const
from scipy import special as sp
from matplotlib import pyplot as plt
from shared_function import coord_trafo


class OAM_profile:
    def __init__(self, wavelength, diameter, l1, l2, p1, p2, x, y, z):

        # set initial values

        self.c = const.c  # speed of light
        self.mu0 = const.mu_0  # absolute permeabillity
        self.eps0 = const.epsilon_0  # absolute permititivity
        self.lamb = wavelength  # wavelength
        self.k = 2 * np.pi / wavelength  # wave_vector
        self.w0 = self.c * self.k  # radial frequency
        self.a = diameter / 2  # radius of the core
        self.l1 = l1
        self.l2 = l2
        self.polarization = np.array(p1)  # polarization state(Jones vector form)
        self.polarization_count = np.array(p2)  # polarization state(Jones vector form)
        self.u_lm = float(sp.jn_zeros(l1, 1))  # First root of l_th order bessel
        self.u_lm_count = float(sp.jn_zeros(l2, 1))
        self.beta = (
            2
            * np.pi
            / self.lamb
            * (1 - 1 / 2 * (self.u_lm * self.lamb / 2 / np.pi / self.a) ** 2)
        )  # propagation constant in HCF

        self.x = x
        self.y = y
        self.z = z

        # some functions to define electric and magnetic field of the beam
        self.u = lambda u_lm, l, r, phi: sp.jve(l, u_lm * r / self.a) * np.exp(
            1j * l * phi
        )
        self.dxu = lambda u_lm, l, r, phi: (
            (
                l / r * sp.jve(l, u_lm * r / self.a)
                - u_lm / self.a * sp.jve(l + 1, u_lm * r / self.a)
            )
            * np.cos(phi)
            - (1j) * l * sp.jve(l, u_lm * r / self.a) * np.sin(phi) / r
        ) * np.exp(1j * l * phi)

        self.dyu = lambda u_lm, l, r, phi: (
            (
                l / r * sp.jve(l, u_lm * r / self.a)
                - u_lm / self.a * sp.jve(l + 1, u_lm * r / self.a)
            )
            * np.sin(phi)
            + (1j) * l * sp.jve(l, u_lm * r / self.a) * np.cos(phi) / r
        ) * np.exp(1j * l * phi)

    def fields(self, direction=1):
        """
        Electric field and magnetic field is calculated in 3D
        This calculation is heavy so 2D version is recommended for plotting purposes
        direction = 1 for the +z propagation and -1 for the counter propagation
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
        Ex_cart = np.zeros(
            (3, len(self.x), len(self.y), len(self.z)), dtype="complex128"
        )
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

        Bx_cart[0, :] = 0
        Bx_cart[1, :] = self.u(u_lm, l, rrr, ppphi)
        Bx_cart[2, :] = 1j / beta * self.dyu(u_lm, l, rrr, ppphi)
        Bx_cart = (1j) * beta * Bx_cart * np.exp(1j * beta * zzz)

        By_cart[0, :] = -self.u(u_lm, l, rrr, ppphi)
        By_cart[1, :] = 0
        By_cart[2, :] = -1j / beta * self.dxu(u_lm, l, rrr, ppphi)
        By_cart = 1j * beta * By_cart * np.exp(1j * beta * zzz)
        return Ex_cart, Ey_cart, Bx_cart, By_cart

    def S_and_I(self):
        # calculate Poynting vector and intensity using the calculated fields
        Ex, Ey, Bx, By = self.fields()
        polarization = self.polarization / np.linalg.norm(self.polarization)
        E_cart = polarization[0] * Ex + polarization[1] * Ey
        B_cart = polarization[0] * Bx + polarization[1] * By

        S = np.real(np.cross(E_cart, np.conjugate(B_cart), axis=0))
        S_cyl = coord_trafo(S, self.ppphi)
        I = np.real(E_cart * np.conjugate(E_cart))
        I = self.c * self.eps0 / 2 * np.sqrt(I[0, :] ** 2 + I[1, :] ** 2 + I[2, :] ** 2)

        self.S = S
        self.S_cyl = S_cyl
        self.I = I

    def standing_S_and_I(self):
        # calculate Poynting vector and intensity of a standing wave using the calculated fields
        Ex, Ey, Bx, By = self.fields()
        Ex_count, Ey_count, Bx_count, By_count = self.fields(direction=-1)
        Ex += Ex_count
        Ey += Ey_count
        Bx += Bx_count
        By += By_count
        polarization = self.polarization / np.linalg.norm(self.polarization)
        polarization2 = self.polarization_count / np.linalg.norm(
            self.polarization_count
        )
        E_cart = polarization[0] * Ex + polarization[1] * Ey
        B_cart = polarization[0] * Bx + polarization[1] * By

        S = np.real(np.cross(E_cart, np.conjugate(B_cart), axis=0))
        S_cyl = coord_trafo(S, self.ppphi)
        I = np.real(E_cart * np.conjugate(E_cart))
        I = self.c * self.eps0 / 2 * np.sqrt(I[0, :] ** 2 + I[1, :] ** 2 + I[2, :] ** 2)

        self.S = S
        self.S_cyl = S_cyl
        self.I = I

    def standing_S_and_I_mix(self):
        # calculate Poynting vector and intensity of a standing wave using the calculated fields
        Ex1, Ey1, Bx1, By1 = self.fields()
        self.l1 = 0
        self.u_lm = float(sp.jn_zeros(0, 1))
        Ex2, Ey2, Bx2, By2 = self.fields()
        phase = 1 * np.pi
        a = 432.5
        b = 227.6
        c = 287
        alpha = c / a
        beta = b / a

        Ex = np.exp(1j * phase) * np.sqrt(alpha) * Ex1 + np.sqrt(beta) * Ex2
        Ey = np.exp(1j * phase) * np.sqrt(alpha) * Ey1 + np.sqrt(beta) * Ey2
        Bx = np.exp(1j * phase) * np.sqrt(alpha) * Bx1 + np.sqrt(beta) * Bx2
        By = np.exp(1j * phase) * np.sqrt(alpha) * By1 + np.sqrt(beta) * By2

        Ex_count, Ey_count, Bx_count, By_count = self.fields(direction=-1)
        Ex += Ex_count
        Ey += Ey_count
        Bx += Bx_count
        By += By_count
        polarization = self.polarization / np.linalg.norm(self.polarization)
        polarization2 = self.polarization_count / np.linalg.norm(
            self.polarization_count
        )
        E_cart = polarization[0] * Ex + polarization[1] * Ey
        B_cart = polarization[0] * Bx + polarization[1] * By

        S = np.real(np.cross(E_cart, np.conjugate(B_cart), axis=0))
        S_cyl = coord_trafo(S, self.ppphi)
        I = np.real(E_cart * np.conjugate(E_cart))
        I = self.c * self.eps0 / 2 * np.sqrt(I[0, :] ** 2 + I[1, :] ** 2 + I[2, :] ** 2)

        self.S = S
        self.S_cyl = S_cyl
        self.I = I

    def save_data(self, location, file_name, group_name):
        """
        save data in h5 format
        x,y,z,S,I are saved for plots
        """
        with h5py.File("{}/{}.h5".format(location, file_name), "a") as hdf:
            try:
                group = hdf.create_group(group_name)
                group.create_dataset("a", data=self.a)
                group.create_dataset("x", data=self.x)
                group.create_dataset("y", data=self.y)
                group.create_dataset("z", data=self.z)
                group.create_dataset("S", data=self.S)
                group.create_dataset("I", data=self.I)

            except:
                del hdf[group_name]
                group = hdf.create_group(group_name)
                group.create_dataset("a", data=self.a)
                group.create_dataset("x", data=self.x)
                group.create_dataset("y", data=self.y)
                group.create_dataset("z", data=self.z)
                group.create_dataset("S", data=self.S)
                group.create_dataset("I", data=self.I)
            hdf.close()
        print("Beam profile data saved")
