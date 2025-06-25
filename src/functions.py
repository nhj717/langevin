import numpy as np
from scipy import constants as const
from scipy import special as sp
from matplotlib import pyplot as plt


def normalize(A):
    for i in range(np.shape(A)[0]):
        for j in range(np.shape(A)[-1]):
            if np.max(abs(A[i, :, :, j])) != 0:
                A[i, :, :, j] = A[i, :, :, j] / np.max(abs(A[:, :, :, j]))

    return A


def coord_trafo(A, theta):
    B = np.zeros_like(A, dtype="complex128")
    B[0, :] = A[0, :] * np.cos(theta) + A[1, :] * np.sin(theta)
    B[1, :] = -A[0, :] * np.sin(theta) + A[1, :] * np.cos(theta)
    B[2, :] = A[2, :]
    return B


def initial_setup(mode_number):
    wavelength = 1.0e-6
    diameter = 40e-6
    return wavelength, diameter, mode_number


class OAM_profile:
    def __init__(self, wavelength, diameter, mode_number):
        # set initial values
        self.c = const.c
        self.mu0 = const.mu_0
        self.eps0 = const.epsilon_0
        self.lamb = wavelength
        self.k = 2 * np.pi / wavelength
        self.w0 = self.c * self.k
        self.a = diameter / 2

        self.l = mode_number
        self.u_lm = float(sp.jn_zeros(self.l, 1))
        self.beta = (
            2
            * np.pi
            / self.lamb
            * (1 - 1 / 2 * (self.u_lm * self.lamb / 2 / np.pi / self.a) ** 2)
        )

        ratio = 1.2
        self.x = np.linspace(-ratio * self.a, ratio * self.a, 100)
        self.y = np.linspace(-ratio * self.a, ratio * self.a, 100)
        self.z = np.linspace(-2 * wavelength, 2 * wavelength, 100)

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

    def fields_xpol(self):
        xxx, yyy, zzz = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        self.xxx = xxx
        self.yyy = yyy
        self.zzz = zzz
        rrr = np.sqrt(xxx**2 + yyy**2)
        ppphi = np.arctan2(yyy, xxx)
        E_cart = np.zeros(
            (3, len(self.x), len(self.y), len(self.z)), dtype="complex128"
        )
        B_cart = np.zeros_like(E_cart, dtype="complex128")

        E_cart[0, :] = self.u(rrr, ppphi)
        E_cart[1, :] = 0
        E_cart[2, :] = 1j / self.beta * self.dxu(rrr, ppphi)
        E_cart = (1j) * self.w0 * E_cart * np.exp(1j * self.beta * zzz)

        B_cart[0, :] = 0
        B_cart[1, :] = self.u(rrr, ppphi)
        B_cart[2, :] = 1j / self.beta * self.dyu(rrr, ppphi)
        B_cart = (1j) * self.beta * B_cart * np.exp(1j * self.beta * zzz)

        S = np.real(np.cross(E_cart, np.conjugate(B_cart), axis=0))
        S_cyl = coord_trafo(S, ppphi)

        I = np.real(E_cart * np.conjugate(E_cart))
        I = np.sqrt(I[0, :] ** 2 + I[1, :] ** 2 + I[2, :] ** 2)

        self.S = S
        self.S_cyl = S_cyl
        self.I = I

    def fields_lcp(self):
        xxx, yyy, zzz = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        self.zzz = zzz
        rrr = np.sqrt(xxx**2 + yyy**2)
        ppphi = np.arctan2(xxx, yyy)
        E_cart = np.zeros(
            (3, len(self.x), len(self.y), len(self.z)), dtype="complex128"
        )
        B_cart = np.zeros_like(E_cart, dtype="complex128")

        E_cart[0, :] = self.u(rrr, ppphi)
        E_cart[1, :] = 1j * self.u(rrr, ppphi)
        E_cart[2, :] = (
            1 / self.beta * (self.dxu(rrr, ppphi) + 1j * self.dyu(rrr, ppphi))
        )
        E_cart = (1j) * self.w0 * E_cart * np.exp(1j * self.beta * zzz)

        B_cart[0, :] = -1j * self.u(rrr, ppphi)
        B_cart[1, :] = self.u(rrr, ppphi)
        B_cart[2, :] = (
            1 / self.beta * (self.dxu(rrr, ppphi) + 1j * self.dyu(rrr, ppphi))
        )
        B_cart = (1j) * self.beta * B_cart * np.exp(1j * self.beta * zzz)

        S = (
            self.c
            * self.eps0
            / 2
            * np.real(np.cross(E_cart, np.conjugate(B_cart), axis=0))
        )
        S_cyl = coord_trafo(S, ppphi)
        I = np.real(E_cart * np.conjugate(E_cart))
        I = self.c * self.eps0 / 2 * np.sqrt(I[0, :] ** 2 + I[1, :] ** 2 + I[2, :] ** 2)

        self.S = S
        self.S_cyl = S_cyl
        self.I = I

    def fields_rcp(self):
        xxx, yyy, zzz = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        self.zzz = zzz
        rrr = np.sqrt(xxx**2 + yyy**2)
        ppphi = np.arctan2(yyy, xxx)
        E_cart = np.zeros(
            (3, len(self.x), len(self.y), len(self.z)), dtype="complex128"
        )
        B_cart = np.zeros_like(E_cart, dtype="complex128")

        E_cart[0, :] = self.u(rrr, ppphi)
        E_cart[1, :] = -1j * self.u(rrr, ppphi)
        E_cart[2, :] = (
            1 / self.beta * (1j * self.dxu(rrr, ppphi) + self.dyu(rrr, ppphi))
        )
        E_cart = (1j) * self.w0 * E_cart * np.exp(1j * self.beta * zzz)

        B_cart[0, :] = 1j * self.u(rrr, ppphi)
        B_cart[1, :] = self.u(rrr, ppphi)
        B_cart[2, :] = (
            1j / self.beta * (1j * self.dxu(rrr, ppphi) + self.dyu(rrr, ppphi))
        )
        B_cart = (1j) * self.beta * B_cart * np.exp(1j * self.beta * zzz)

        S = (
            self.c
            * self.eps0
            / 2
            * np.real(np.cross(E_cart, np.conjugate(B_cart), axis=0))
        )
        S_cyl = coord_trafo(S, ppphi)
        I = np.real(E_cart * np.conjugate(E_cart))
        I = self.c * self.eps0 / 2 * np.sqrt(I[0, :] ** 2 + I[1, :] ** 2 + I[2, :] ** 2)

        self.S = S
        self.S_cyl = S_cyl
        self.I = I

    def fields_ypol(self):
        xxx, yyy, zzz = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        self.zzz = zzz
        rrr = np.sqrt(xxx**2 + yyy**2)
        ppphi = np.arctan2(yyy, xxx)
        E_cart = np.zeros(
            (3, len(self.x), len(self.y), len(self.z)), dtype="complex128"
        )
        B_cart = np.zeros_like(E_cart, dtype="complex128")

        E_cart[0, :] = 0
        E_cart[1, :] = self.u(rrr, ppphi)
        E_cart[2, :] = 1j / self.beta * self.dyu(rrr, ppphi)
        E_cart = (1j) * self.w0 * E_cart * np.exp(1j * self.beta * zzz)

        B_cart[0, :] = -1j * self.beta * self.u(rrr, ppphi)
        B_cart[1, :] = 0
        B_cart[2, :] = -self.dxu(rrr, ppphi)
        B_cart = B_cart * np.exp(1j * self.beta * zzz)

        S = np.real(np.cross(E_cart, np.conjugate(B_cart), axis=0))
        S_cyl = coord_trafo(S, ppphi)
        I = np.real(E_cart * np.conjugate(E_cart))
        I = self.c * self.eps0 / 2 * np.sqrt(I[0, :] ** 2 + I[1, :] ** 2 + I[2, :] ** 2)

        self.S = S
        self.S_cyl = S_cyl
        self.I = I

    def plot_field_dist(self):

        S = self.S
        I = self.I
        xx, yy = np.meshgrid(self.x, self.y, indexing="ij")
        step = 5

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        fig.suptitle(f"Poynting vector of OAM")

        circle = plt.Circle(
            (0, 0), self.a, fill=False, color="black", linewidth=1, linestyle="--"
        )
        z_target = 0
        zi = (np.abs(self.z - z_target)).argmin()

        S_theta_max = np.max(abs(self.S_cyl[1, :, :, zi]))
        print(f"Value of S_theta is {S_theta_max*1E-19} a.u.")

        I_zi = I[:, :, zi] / np.max(I[:, :, zi])

        pcm = ax.pcolor(xx, yy, I_zi, cmap="jet")
        cb = plt.colorbar(pcm, shrink=0.75)
        vec = ax.quiver(
            xx[::step, ::step],
            yy[::step, ::step],
            S[0, ::step, ::step, zi],
            S[1, ::step, ::step, zi],
            color="white",
            # scale=15,
        )
        ax.set(aspect="equal")
        ax.add_patch(circle)

        plt.show(block=True)
