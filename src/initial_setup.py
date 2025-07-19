# Initial setup of different calculations are triggered from here
import numpy as np


def beam_profile_initial_setup_3D(mode_number, polarization):
    wavelength = 1.064e-6  # laser wavelength
    diameter = 44e-6  # diamber of the fiber core
    ratio = 1.2  # ratio for the range
    N = 100  # number of points in  the array
    x = np.linspace(-ratio * diameter / 2, ratio * diameter / 2, N)
    y = np.linspace(-ratio * diameter / 2, ratio * diameter / 2, N)
    z = np.linspace(-2 * wavelength, 2 * wavelength, N)
    return wavelength, diameter, mode_number, polarization, x, y, z


def beam_profile_initial_setup_XY(mode_number, polarization):
    wavelength = 1.064e-6  # laser wavelength
    diameter = 44e-6  # diamber of the fiber core
    ratio = 1.2  # ratio for the range
    N = 1000  # number of points in  the array
    x = np.linspace(-ratio * diameter / 2, ratio * diameter / 2, N)
    y = np.linspace(-ratio * diameter / 2, ratio * diameter / 2, N)
    z = np.linspace(0, 0, 1)
    return wavelength, diameter, mode_number, polarization, x, y, z


def beam_profile_initial_setup_XZ(mode_number, polarization):
    wavelength = 1.064e-6  # laser wavelength
    diameter = 44e-6  # diamber of the fiber core
    ratio = 1.2  # ratio for the range
    N = 1000  # number of points in  the array
    x = np.linspace(-ratio * diameter / 2, ratio * diameter / 2, N)
    y = np.linspace(0, 0, 1)
    z = np.linspace(-2 * wavelength, 2 * wavelength, N)
    return wavelength, diameter, mode_number, polarization, x, y, z


def oam_trapping_initial_setup():
    diameter = 400  # particle size in nanometers
    eps_glass = 3.9  # relative permitivity of the glass
    power = 200  # laser power in mW from both sides
    pressure = 1000  # surrounding pressure in mbar
    core_radius = 22  # fiber core radius in um
    N = int(1e8)  # Total number of sampling
    delt = 1e-6  # in seconds, time resolution of the simulation
    iteration = 10  # number of iterations that are averaged
    mode_number = 1
    return (
        diameter,
        eps_glass,
        power,
        pressure,
        core_radius,
        N,
        delt,
        iteration,
        mode_number,
    )
