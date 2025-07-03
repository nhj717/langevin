from beam_profile import *
from langevin_eq import *
import langevin_averaged as langav
from oam_langevin import *


def test_field_distribution():
    sim = OAM_profile(*initial_setup(1, [1, -1j]))
    sim.fields()
    # sim.S_and_I()
    sim.standing_S_and_I()
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
    sim.plot_x()
    sim.plot_y()
    sim.plot_z()
    sim.plot_spectrums()


def test_oam_langevin():
    sim = oam_Langevin(1)
    sim.langevin_eq()
