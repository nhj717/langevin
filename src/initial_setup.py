#Initial setup of different calculations are triggered from here

def beam_profile_initial_setup(mode_number, polarization):
    wavelength = 1.064e-6
    diameter = 44e-6
    return wavelength, diameter, mode_number, polarization


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