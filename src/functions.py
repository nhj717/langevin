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
    B[0, :] = A[0, :] * np.cos(theta) - A[1, :] * np.sin(theta)
    B[1, :] = A[0, :] * np.sin(theta) + A[1, :] * np.cos(theta)
    B[2, :] = A[2, :]
    return B


def initial_setup(mode_number):
    wavelength = 1.0
    diameter = 40
    return wavelength, diameter, mode_number


class OAM_profile:
    def __init__(self, wavelength, diameter, mode_number):
        # set initial values
        self.c = const.c
        self.mu0 = const.mu_0
        self.eps0 = const.epsilon_0
        self.lamb = wavelength  # micrometer
        self.k = 2 * np.pi / wavelength
        self.w0 = self.c * self.k * 1e6
        self.a = diameter / 2  # micrometer

        self.l = mode_number
        self.u_lm = float(sp.jn_zeros(self.l, 3)[0])
        self.beta = (
            2
            * np.pi
            / self.lamb
            * (1 - 1 / 2 * (self.u_lm * self.lamb / 2 / np.pi / self.a) ** 2)
        )

        self.x = np.linspace(-2 * self.a, 2 * self.a, 100)
        self.y = np.linspace(-2 * self.a, 2 * self.a, 100)
        self.z = np.linspace(-2 * wavelength, 2 * wavelength, 100)

        self.u = lambda r, phi: sp.jve(self.l, r) * np.exp(1j * self.l * phi)
        self.dxu = lambda r, phi: sp.jve(self.l, r) * (
            (
                self.l / r * sp.jve(self.l, r)
                - self.u_lm / self.a * sp.jve(self.l + 1, r)
            )
            * np.cos(phi)
            - (1j) * self.l * sp.jve(self.l, r) * np.sin(phi) / r
        )
        self.dyu = lambda r, phi: sp.jve(self.l, r) * (
            (
                self.l / r * sp.jve(self.l, r)
                - self.u_lm / self.a * sp.jve(self.l + 1, r)
            )
            * np.sin(phi)
            + (1j) * self.l * sp.jve(self.l, r) * np.cos(phi) / r
        )

    def fields(self):
        xxx, yyy, zzz = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        rrr = np.sqrt(xxx**2 + yyy**2)
        ppphi = np.arctan2(xxx, yyy)
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

        S = np.real(np.cross(E_cart, B_cart, axis=0))
        I = np.real(E_cart * np.conjugate(E_cart))
        I = np.sqrt(I[0, :] ** 2 + I[1, :] ** 2 + I[2, :] ** 2)

        self.S = S
        self.I = I

    def plot_field_dist(self):

        S = self.S
        I = self.I
        xx, yy = np.meshgrid(self.x, self.y)
        step = 5

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        fig.suptitle(f"Poynting vector of OAM")

        circle = plt.Circle(
            (0, 0), self.a, fill=False, color="black", linewidth=1, linestyle="--"
        )

        zi = int(len(self.z) / 2)

        pcm = ax.pcolor(xx, yy, I[:, :, zi], cmap="jet")
        cb = plt.colorbar(pcm, shrink=0.75)
        vec = ax.quiver(
            xx[::step, ::step],
            yy[::step, ::step],
            S[0, ::step, ::step, zi],
            S[1, ::step, ::step, zi],
            color="white",
            scale=15,
        )
        ax.set(aspect="equal")
        ax.add_patch(circle)

        plt.show(block=True)
