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
    def __init__(self, wavelength, diameter, mode_number, polarization,x,y,z):

        # set initial values

        self.c = const.c                                #speed of light
        self.mu0 = const.mu_0                           #absolute permeabillity
        self.eps0 = const.epsilon_0                     #absolute permititivity
        self.lamb = wavelength                          #wavelength
        self.k = 2 * np.pi / wavelength                 #wave_vector
        self.w0 = self.c * self.k                       #radial frequency
        self.a = diameter / 2                           #radius of the core
        self.polarization = np.array(polarization)      #polarization state(Jones vector form)
        self.l = mode_number                            #mode number
        self.u_lm = float(sp.jn_zeros(self.l, 1))   #First root of l_th order bessel
        self.beta = (
            2
            * np.pi
            / self.lamb
            * (1 - 1 / 2 * (self.u_lm * self.lamb / 2 / np.pi / self.a) ** 2)
        )                                               #propagation constant in HCF

        self.x = x
        self.y = y
        self.z = z


        #some functions to define electric and magnetic field of the beam
        self.u = lambda r, phi: sp.jve(self.l, self.u_lm * r / self.a) * np.exp(
            1j * self.l * phi
        )
        self.dxu = lambda r, phi: (
            (
                self.l / r * sp.jve(self.l, self.u_lm * r / self.a)
                - self.u_lm / self.a * sp.jve(self.l + 1, self.u_lm * r / self.a)
            )
            * np.cos(phi)
            - (1j) * self.l * sp.jve(self.l, self.u_lm * r / self.a) * np.sin(phi) / r
        ) * np.exp(1j * self.l * phi)

        self.dyu = lambda r, phi: (
            (
                self.l / r * sp.jve(self.l, self.u_lm * r / self.a)
                - self.u_lm / self.a * sp.jve(self.l + 1, self.u_lm * r / self.a)
            )
            * np.sin(phi)
            + (1j) * self.l * sp.jve(self.l, self.u_lm * r / self.a) * np.cos(phi) / r
        ) * np.exp(1j * self.l * phi)

    def fields(self,direction = 1):
        """
        Electric field and magnetic field is calculated in 3D
        This calculation is heavy so 2D version is recommended for plotting purposes
        direction = 1 for the +z propagation and -1 for the counter propagation
        """
        beta = direction*self.beta
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

        Ex_cart[0, :] = self.u(rrr, ppphi)
        Ex_cart[1, :] = 0
        Ex_cart[2, :] = 1j / beta * self.dxu(rrr, ppphi)
        Ex_cart = (1j) * self.w0 * Ex_cart * np.exp(1j * beta * zzz)

        Ey_cart[0, :] = 0
        Ey_cart[1, :] = self.u(rrr, ppphi)
        Ey_cart[2, :] = 1j / beta * self.dyu(rrr, ppphi)
        Ey_cart = (1j) * self.w0 * Ey_cart * np.exp(1j * beta * zzz)

        Bx_cart[0, :] = 0
        Bx_cart[1, :] = self.u(rrr, ppphi)
        Bx_cart[2, :] = 1j / beta * self.dyu(rrr, ppphi)
        Bx_cart = (1j) * beta * Bx_cart * np.exp(1j * beta * zzz)

        By_cart[0, :] = -self.u(rrr, ppphi)
        By_cart[1, :] = 0
        By_cart[2, :] = -1j / beta * self.dxu(rrr, ppphi)
        By_cart = 1j * beta * By_cart * np.exp(1j * beta * zzz)
        return Ex_cart, Ey_cart, Bx_cart, By_cart

    def S_and_I(self):
        #calculate Poynting vector and intensity using the calculated fields
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
                group.create_dataset('a',data=self.a)
                group.create_dataset('x',data = self.x)
                group.create_dataset('y', data=self.y)
                group.create_dataset('z', data=self.z)
                group.create_dataset("S", data=self.S)
                group.create_dataset("I", data=self.I)

            except:
                del hdf[group_name]
                group = hdf.create_group(group_name)
                group.create_dataset('a', data=self.a)
                group.create_dataset('x', data=self.x)
                group.create_dataset('y', data=self.y)
                group.create_dataset('z', data=self.z)
                group.create_dataset("S", data=self.S)
                group.create_dataset("I", data=self.I)
            hdf.close()
        print("Beam profile data saved")
