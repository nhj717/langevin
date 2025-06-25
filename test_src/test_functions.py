from functions import *


def test_field_distribution():
    sim = OAM_profile(*initial_setup(1))
    sim.fields_xpol()
    sim.plot_field_dist()
