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
    I_max_index = np.argmax(I[int(np.size(self.x) / 2), :, zi])
    S_r_max = abs(self.S_cyl[0, int(np.size(self.x) / 2), I_max_index, zi])
    S_theta_max = abs(self.S_cyl[1, int(np.size(self.x) / 2), I_max_index, zi])
    print(f"Value of S_r is {S_r_max * 1E-19} a.u.")
    print(f"Value of S_theta is {S_theta_max*1E-19} a.u.")

    I_zi = I[:, :, zi] / np.max(I[:, :, zi])
    circle2 = plt.Circle(
        (0, self.y[I_max_index]),
        self.a / 50,
        fill=True,
        color="black",
        linewidth=1,
        linestyle="--",
    )

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
    ax.add_patch(circle2)

    plt.show(block=True)


def surface_plot(self):
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
