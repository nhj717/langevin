from beam_profile import *
from langevin_eq import *
import langevin_averaged as langav
import oam_langevin as oamlan


def test_field_distribution():
    sim = OAM_profile(*initial_setup(1, [1, 1j]))
    sim.fields()
    sim.S_and_I()
    # sim.standing_S_and_I()
    sim.plot_field_dist()


def test_langevin_eq():
    sim = Langevin()
    sim.langevin_eq()
    sim.plot_x()
    sim.plot_z()
    sim.plot_spectrums()


def test_langevin_averaged():
    sim = langav.Langevin_averaged(*langav.initial_setup())
    sim.langevin_eq()
    sim.plot("x")
    sim.plot("y")
    sim.plot("z")
    sim.plot_spectrums()
    sim.plot_summed_spectrum()


def test_oam_langevin():
    sim = oamlan.oam_Langevin(*oamlan.initial_setup())
    sim.langevin_eq()
