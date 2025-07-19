"""
All of the functions for plotting and saving purpose
"""

import numpy as np
import scipy.fft as fft
from scipy.optimize import curve_fit
import shared_function
import matplotlib.pyplot as plt


def plot_XY_with_Poynting(location, file_name, group_name):
    # Load data from the saved h5 file
    data_label, data = shared_function.read_data(location, file_name, group_name)
    a, x, y, z, S, I = (
        data[data_label.index("a")],
        data[data_label.index("x")],
        data[data_label.index("y")],
        data[data_label.index("z")],
        data[data_label.index("S")],
        data[data_label.index("I")],
    )
    xx, yy = np.meshgrid(x * 1e6, y * 1e6, indexing="ij")
    step = 50

    fig, ax = plt.subplots(1, 1, figsize=(7, 6), tight_layout=True)

    # Circle represents the fiber core
    circle = plt.Circle(
        (0, 0), a * 1e6, fill=False, color="black", linewidth=1, linestyle="--"
    )
    # Normalize intensity
    I_norm = I[:, :, 0] / np.max(I[:, :, 0])

    # 2D colormap of the intensity
    pcm = ax.pcolor(xx, yy, I_norm, cmap="jet")
    cb = plt.colorbar(pcm, shrink=0.92)

    # Quiver(vector) plot of the poynting vector
    vec = ax.quiver(
        xx[::step, ::step],
        yy[::step, ::step],
        S[0, ::step, ::step, 0],
        S[1, ::step, ::step, 0],
        color="white",
        scale=1e21,
    )
    ax.set(aspect="equal")
    ax.add_patch(circle)
    # plt.xlim(-20, 20)
    # plt.ylim(-20, 20)
    plt.title("Norm. Intensity in XY")
    plt.xlabel(r"x [$\mu m$]")
    plt.ylabel(r"y [$\mu m$]")
    plt.show(block=True)
    return fig


def plot_XY(location, file_name, group_name):
    # Load data from the saved h5 file
    data_label, data = shared_function.read_data(location, file_name, group_name)
    a, x, y, z, S, I = (
        data[data_label.index("a")],
        data[data_label.index("x")],
        data[data_label.index("y")],
        data[data_label.index("z")],
        data[data_label.index("S")],
        data[data_label.index("I")],
    )
    xx, yy = np.meshgrid(x * 1e6, y * 1e6, indexing="ij")

    fig, ax = plt.subplots(1, 1, figsize=(7, 6), tight_layout=True)

    # Circle represents the fiber core
    circle = plt.Circle(
        (0, 0), a * 1e6, fill=False, color="black", linewidth=1, linestyle="--"
    )
    # Normalize intensity
    I_norm = I[:, :, 0] / np.max(I[:, :, 0])

    # 2D colormap of the intensity
    pcm = ax.pcolor(xx, yy, I_norm, cmap="jet")
    cb = plt.colorbar(pcm, shrink=0.92)

    ax.set(aspect="equal")
    ax.add_patch(circle)
    # plt.xlim(-20, 20)
    # plt.ylim(-20, 20)
    plt.title("Norm. Intensity in XY")
    plt.xlabel(r"x [$\mu m$]")
    plt.ylabel(r"y [$\mu m$]")
    plt.show(block=True)
    return fig


def plot_XZ(location, file_name, group_name):
    # Load data from the saved h5 file
    data_label, data = shared_function.read_data(location, file_name, group_name)
    a, x, y, z, S, I = (
        data[data_label.index("a")],
        data[data_label.index("x")],
        data[data_label.index("y")],
        data[data_label.index("z")],
        data[data_label.index("S")],
        data[data_label.index("I")],
    )
    zz, xx = np.meshgrid(z * 1e6, x * 1e6, indexing="ij")
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), tight_layout=True)
    pcm = ax.pcolor(zz, xx, I[:, 0, :].T / I[:, 0, :].max(), cmap="jet")
    cb = plt.colorbar(pcm, shrink=0.6)
    ax.set_box_aspect(0.5)
    plt.xlim(-1.2, 1.2)
    plt.ylim(-20, 20)
    plt.title("Norm. Intensity in XZ")
    plt.xlabel(r"z [$\mu m$]")
    plt.ylabel(r"x [$\mu m$]")
    plt.show(block=True)
    return fig


def plot_package(m, N, t, f, x, v, xyz):
    if xyz == "x":
        index = 0
    elif xyz == "y":
        index = 1
    else:
        index = 2

    # plots the
    fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)
    ax1.plot(t, x[index, :] * 1e9)
    plt.xlabel("Time [s]")
    plt.ylabel(f"{xyz} [$nm$]")
    plt.xlim(0, 0.05)
    plt.show(block=True)

    fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)
    ax2.plot(x[index, :] * 1e9, m * v[index, :] * 1e21)
    plt.xlabel(f"{xyz} [$nm$]")
    plt.ylabel("P [$fg*um/s$]")
    plt.show(block=True)

    fig3, ax3 = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)
    x_fft = 2.0 / N * fft.fft(x[index, :])[: int(N / 2)]
    x_fft = abs(x_fft) ** 2
    ax3.plot(f * 1e-3, np.log10(x_fft))
    index = 2 * int(x_fft.argmax())
    if index > np.size(f):
        plt.xlim(0.2, f[-1] * 1e-3)
    elif f[index] * 1e-3 < 5:
        plt.xlim(0.2, 5)
    else:
        plt.xlim(0.2, f[index] * 1e-3)
    plt.xlabel("f [kHz]")
    plt.ylabel("$log_{10}S$ [a.u.]")
    plt.show(block=True)

    return fig1, fig2, fig3


def plot_particle_xy(x):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)
    ax.plot(x[0, :], x[1, :])
    plt.xlim(-1.5e-5, 1.5e-5)
    plt.ylim(-1.5e-5, 1.5e-5)
    plt.title("Particle Position in XY plane")
    plt.xlabel(r"x [$\mu m$]")
    plt.ylabel(r"Y [$\mu m$]")
    plt.show(block=True)
    return fig


def plot_spectrums(N, gamma0, f, x):
    omega = 2 * np.pi * f
    f_start = int(np.abs(f - 50).argmin())
    x_fft = 2.0 / N * fft.fft(x[0, :])[: int(N / 2)]
    x_fft = abs(x_fft) ** 2
    peak_w_x = omega[np.argmax(x_fft)]
    lorentzian_fit_coeff, lorentzian_fit_error = curve_fit(
        shared_function.lorentzian, omega, x_fft, p0=[peak_w_x, 5e-6, gamma0]
    )
    x_fft_fit = shared_function.lorentzian(
        omega,
        lorentzian_fit_coeff[0],
        lorentzian_fit_coeff[1],
        lorentzian_fit_coeff[2],
    )
    y_fft = 2.0 / N * fft.fft(x[1, :])[: int(N / 2)]
    y_fft = abs(y_fft) ** 2
    peak_w_y = omega[np.argmax(y_fft[f_start:])]
    lorentzian_fit_coeff1, lorentzian_fit_error1 = curve_fit(
        shared_function.lorentzian,
        omega[f_start:],
        y_fft[f_start:],
        p0=[peak_w_y, 5e-6, gamma0],
    )
    y_fft_fit = shared_function.lorentzian(
        omega,
        lorentzian_fit_coeff1[0],
        lorentzian_fit_coeff1[1],
        lorentzian_fit_coeff1[2],
    )
    z_fft = 2.0 / N * fft.fft(x[2, :])[: int(N / 2)]
    z_fft = abs(z_fft) ** 2
    peak_w_z = omega[np.argmax(z_fft)]
    lorentzian_fit_coeff2, lorentzian_fit_error2 = curve_fit(
        shared_function.lorentzian, omega, z_fft, p0=[peak_w_z, 5e-6, gamma0]
    )
    z_fft_fit = shared_function.lorentzian(
        omega,
        lorentzian_fit_coeff2[0],
        lorentzian_fit_coeff2[1],
        lorentzian_fit_coeff2[2],
    )

    print(
        f"Actual gamma0 is {gamma0 / (2 * np.pi)}Hz and the calculated gamma0 is {lorentzian_fit_coeff[2] / (2 * np.pi)}Hz for x, {lorentzian_fit_coeff1[2] / (2 * np.pi)}Hz for y and {lorentzian_fit_coeff2[2] / (2 * np.pi)}Hz for z"
    )
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), tight_layout=True)
    ax.plot(f * 1e-3, np.log10(x_fft), "orange", label="xfft")
    ax.plot(f * 1e-3, np.log10(x_fft_fit), "red", label="xfft fit")
    ax.plot(f * 1e-3, np.log10(y_fft), "cyan", label="yfft")
    ax.plot(f * 1e-3, np.log10(y_fft_fit), "blue", label="yfft fit")
    ax.plot(f * 1e-3, np.log10(z_fft), "brown", label="zfft")
    ax.plot(f * 1e-3, np.log10(z_fft_fit), "black", label="zfft fit")
    plt.xlim(0.1, 100)
    plt.xlabel("f [kHz]")
    plt.ylabel("S [a.u.]")
    plt.legend()
    plt.show(block=True)
    return fig


def plot_summed_spectrum(N, f, x):
    x_fft = 2.0 / N * fft.fft(x[0, :])[: int(N / 2)]
    x_fft = abs(x_fft) ** 2
    y_fft = 2.0 / N * fft.fft(x[1, :])[: int(N / 2)]
    y_fft = abs(y_fft) ** 2
    z_fft = 2.0 / N * fft.fft(x[2, :])[: int(N / 2)]
    z_fft = abs(z_fft) ** 2
    total_fft = x_fft + z_fft

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), tight_layout=True)
    ax.plot(f * 1e-3, np.log10(total_fft), "black")
    plt.xlim(0.1, 100)
    plt.xlabel("f [kHz]")
    plt.ylabel("S [a.u.]")
    plt.show(block=True)
    return fig


def save_figure(
    fig,
    file_name,
    location="/Users/hnam/pycharm_projects/langevin/figures",
    dpi_value=300,
):
    image_format = "png"  # e.g .png, .svg, etc.
    fig.savefig(f"{location}/{file_name}.png", format=image_format, dpi=dpi_value)
    print("Image saved")
