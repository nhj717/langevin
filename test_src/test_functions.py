from beam_profile import *
from langevin_eq import *


def test_field_distribution():
    sim = OAM_profile(*initial_setup(1, [1, -1j]))
    sim.fields()
    # sim.S_and_I()
    sim.standing_S_and_I()
    sim.plot_field_dist()


def test_langevin_eq():
    sim = Langevin()
    sim.langevin_eq()
    # sim.plot()
    sim.plot_z()
