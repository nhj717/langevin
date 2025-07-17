"""
All of the functions for plotting and saving purpose
"""

import numpy as np
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
    step = 5

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

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
        # scale=15,
    )
    ax.set(aspect="equal")
    ax.add_patch(circle)
    # plt.xlim(-20, 20)
    # plt.ylim(-20, 20)
    plt.title("Norm. Intensity in XY")
    plt.xlabel(r"x [um]")
    plt.ylabel(r"y [um]")
    plt.tight_layout()
    plt.show(block=True)


def plot_XZ(self):
    I = self.I
    yc = int(np.size(self.z) / 2)
    zc = int(np.size(self.z) / 2)
    xx, yy, zz = np.meshgrid(self.x, self.y, self.z, indexing="ij")
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    pcm = ax.pcolor(
        1e6 * zz[:, yc, :], 1e6 * xx[:, yc, :], I[:, yc, :] / I.max(), cmap="jet"
    )
    cb = plt.colorbar(pcm, shrink=0.4)
    ax.set_box_aspect(0.5)
    plt.xlim(-1.2, 1.2)
    plt.ylim(-20, 20)
    plt.title("Intensity in XZ Plane")
    plt.xlabel(r"z [um]")
    plt.ylabel(r"x [um]")
    plt.show(block=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    pcm = ax.pcolor(
        1e6 * xx[:, :, zc], 1e6 * yy[:, :, zc], I[:, :, zc] / I.max(), cmap="jet"
    )
    cb = plt.colorbar(pcm, shrink=0.75)
    ax.set(aspect="equal")
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.title("Intensity in XZ Plane")
    plt.xlabel(r"x [um]")
    plt.ylabel(r"y [um]")
    plt.show(block=True)
