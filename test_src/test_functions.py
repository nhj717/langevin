from functions import *


def test_field_distribution():
    sim = OAM_profile(*initial_setup(1, [1, 0]))
    sim.fields()
    sim.S_and_I()
    # sim.standing_S_and_I()
    sim.plot_field_dist()
