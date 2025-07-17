# Initial setup of different calculations are triggered from here
import numpy as np


def beam_profile_initial_setup_3D(mode_number, polarization):
    wavelength = 1.064e-6
    diameter = 44e-6
    ratio = 1.2
    N = 100
    x = np.linspace(-ratio * diameter / 2, ratio * diameter / 2, N)
    y = np.linspace(-ratio * diameter / 2, ratio * diameter / 2, N)
    z = np.linspace(-2 * wavelength, 2 * wavelength, N)
    return wavelength, diameter, mode_number, polarization, x, y, z


def beam_profile_initial_setup_XY(mode_number, polarization):
    wavelength = 1.064e-6
    diameter = 44e-6
    ratio = 1.2
    N = 1000
    x = np.linspace(-ratio * diameter / 2, ratio * diameter / 2, N)
    y = np.linspace(-ratio * diameter / 2, ratio * diameter / 2, N)
    z = np.linspace(0, 0, 1)
    return wavelength, diameter, mode_number, polarization, x, y, z


def beam_profile_initial_setup_XZ(mode_number, polarization):
    wavelength = 1.064e-6
    diameter = 44e-6
    ratio = 1.2
    N = 1000
    x = np.linspace(-ratio * diameter / 2, ratio * diameter / 2, N)
    y = np.linspace(0, 0, 1)
    z = np.linspace(-2 * wavelength, 2 * wavelength, N)
    return wavelength, diameter, mode_number, polarization, x, y, z


def oam_trapping_initial_setup():
    diameter = 400  # in nanometers
    eps_glass = 3.9
    power = 400  # in mW from both sides
    pressure = 1000  # in mbar
    core_radius = 22  # in um
    N = int(1e5)  # Total number of sampling
    delt = 1e-6  # in seconds, time resolution of the simulation
    iteration = 10  # number of sampling
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
