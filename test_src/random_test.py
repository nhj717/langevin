import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

x0, x1 = 0, 1
y0, y1 = 0, 1
z0, z1 = 0, 1
cut_size = 0.5
xc0, xc1 = x0, x1
yc0, yc1 = y0, y1 - cut_size
zc0, zc1 = z1 - cut_size, z1


def func(x, y):
    return np.sin(np.pi * x) * np.cos(np.pi * y)


def plot_function_face(x_range, y_range, fixed_val, orientation="z"):
    x = np.linspace(*x_range, 50)
    y = np.linspace(*y_range, 50)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    norm = (Z - Z.min()) / (Z.max() - Z.min())
    facecolors = plt.cm.viridis(norm)

    if orientation == "z":
        ax.plot_surface(
            X,
            Y,
            np.full_like(Z, fixed_val),
            facecolors=facecolors,
            rstride=1,
            cstride=1,
        )
    elif orientation == "x":
        ax.plot_surface(
            np.full_like(Z, fixed_val),
            X,
            Y,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
        )
    elif orientation == "y":
        ax.plot_surface(
            X,
            np.full_like(Z, fixed_val),
            Y,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
        )


# Add functional faces (see earlier logic)
# Bottom, left, front
plot_function_face((x0, x1), (y0, y1), z0, "z")
plot_function_face((x0, x1), (z0, z1), y1, "y")
# Top (with cut-out), right, back (split sections)
plot_function_face((x0, x1), (yc1, y1), z1, "z")
plot_function_face((x0, x1), (yc0, yc1), zc0, "z")
plot_function_face((yc1, y1), (z0, z1), x1, "x")
plot_function_face((yc0, yc1), (z0, zc0), x1, "x")
plot_function_face((yc1, y1), (z0, z1), x0, "x")
plot_function_face((yc0, yc1), (z0, zc0), x0, "x")
plot_function_face((x0, x1), (z0, zc0), y0, "y")
plot_function_face((x0, x1), (zc0, zc1), yc1, "y")


# Interior gray cut walls
def plot_flat_face(x_range, y_range, z_fixed, orientation="z"):
    x = np.linspace(*x_range, 2)
    y = np.linspace(*y_range, 2)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z_fixed)
    if orientation == "z":
        ax.plot_surface(X, Y, Z, color="gray", alpha=0.5)
    elif orientation == "x":
        ax.plot_surface(Z, X, Y, color="gray", alpha=0.5)
    elif orientation == "y":
        ax.plot_surface(X, Z, Y, color="gray", alpha=0.5)


plot_flat_face((xc0, xc1), (yc0, yc1), zc0, "z")
plot_flat_face((xc0, xc1), (zc0, zc1), yc1, "y")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Box with Function-Based Surfaces and Cut-Out Corner")
ax.set_box_aspect([1, 2, 1])
plt.tight_layout()
plt.show()
