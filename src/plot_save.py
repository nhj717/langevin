"""
All of the functions for plotting and saving purpose
"""

import numpy as np
from matplotlib.pyplot import tight_layout

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
    plt.tight_layout()
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


def save_figure(
    fig,
    file_name,
    location="/Users/hnam/pycharm_projects/langevin/figures",
    dpi_value=300,
):
    image_format = "png"  # e.g .png, .svg, etc.
    fig.savefig(f"{location}/{file_name}.png", format=image_format, dpi=dpi_value)
    print("Image saved")
