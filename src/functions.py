import numpy as np
from scipy import constants as const
from scipy import special as sp
from matplotlib import pyplot as plt


def coord_trafo(A, theta):
    B = np.zeros_like(A, dtype="complex128")
    B[0, :] = A[0, :] * np.cos(theta) + A[1, :] * np.sin(theta)
    B[1, :] = -A[0, :] * np.sin(theta) + A[1, :] * np.cos(theta)
    B[2, :] = A[2, :]
    return B


def initial_setup(mode_number, polarization):
    wavelength = 1.0e-6
    diameter = 40e-6
    return wavelength, diameter, mode_number, polarization


class OAM_profile:
    def __init__(self, wavelength, diameter, mode_number, polarization):
        # set initial values
        self.c = const.c
        self.mu0 = const.mu_0
        self.eps0 = const.epsilon_0
        self.lamb = wavelength
        self.k = 2 * np.pi / wavelength
        self.w0 = self.c * self.k
        self.a = diameter / 2
        self.polarization = np.array(polarization)
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

    def fields(self):
        beta = self.beta
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
        Ex, Ey, Bx, By = self.fields()
        polarization = self.polarization / np.linalg.norm(self.polarization)
        print(polarization[0])
        print(polarization[1])
        E_cart = polarization[0] * Ex + polarization[1] * Ey
        B_cart = polarization[0] * Bx + polarization[1] * By

        S = np.real(np.cross(E_cart, np.conjugate(B_cart), axis=0))
        S_cyl = coord_trafo(S, self.ppphi)
        I = np.real(E_cart * np.conjugate(E_cart))
        I = self.c * self.eps0 / 2 * np.sqrt(I[0, :] ** 2 + I[1, :] ** 2 + I[2, :] ** 2)

        self.S = S
        self.S_cyl = S_cyl
        self.I = I

    def counter_fields(self):
        beta = -self.beta
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

    def standing_S_and_I(self):
        Ex, Ey, Bx, By = self.fields()
        Ex_count, Ey_count, Bx_count, By_count = self.counter_fields()
        Ex += Ex_count
        Ey += Ey_count
        Bx += Bx_count
        By += By_count
        polarization = self.polarization / np.linalg.norm(self.polarization)
        print(polarization[0])
        print(polarization[1])
        E_cart = polarization[0] * Ex + polarization[1] * Ey
        B_cart = polarization[0] * Bx + polarization[1] * By

        S = np.real(np.cross(E_cart, np.conjugate(B_cart), axis=0))
        S_cyl = coord_trafo(S, self.ppphi)
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
        z_target = 0 * self.lamb / 8
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
