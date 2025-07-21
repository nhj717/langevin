# Shared functions are stored here

import numpy as np
import h5py


def coord_trafo(A, theta):
    B = np.zeros_like(A, dtype="complex128")
    B[0, :] = A[0, :] * np.cos(theta) + A[1, :] * np.sin(theta)
    B[1, :] = -A[0, :] * np.sin(theta) + A[1, :] * np.cos(theta)
    B[2, :] = A[2, :]
    return B


def lorentzian(x, x0, a, gamma):
    x0 = 2 * np.pi * x0
    x = 2 * np.pi * x
    gamma = 2 * np.pi * gamma
    return a * gamma / ((x0**2 - x**2) ** 2 + (x * gamma) ** 2)


def read_data(location, file_name, group_name):
    df = h5py.File("{}/{}.h5".format(location, file_name), "r")
    try:
        data_label = list(df[group_name].keys())
        data = []
        for name in data_label:
            data.append(np.array(df[group_name][name]))

    except:
        data_label = "data"
        data = np.array(df[group_name])

    df.close()
    return data_label, data


def check_data(location, file_name):
    df = h5py.File("{}/{}.h5".format(location, file_name), "r")
    names = list(df.keys())
    print(*names, sep="\n")
