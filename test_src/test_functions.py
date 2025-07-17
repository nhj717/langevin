import initial_setup as inset
from beam_profile import *
from langevin_eq import *
import langevin_averaged as langav
import oam_langevin as oamlan
import plot_save


def test_field_distribution():
    sim = OAM_profile(*inset.beam_profile_initial_setup_XZ(1, [1, 1j]))
    # sim.S_and_I()
    sim.standing_S_and_I()
    location = "/Users/hnam/pycharm_projects/langevin/data"
    folder_name = "beam_profile_data"
    group_name = "test"
    sim.save_data(location, folder_name, group_name)


def test_plot_save():
    location = "/Users/hnam/pycharm_projects/langevin/data"
    file_name = "beam_profile_data"
    group_name = "test"
    plot_save.plot_XZ(location, file_name, group_name)


def test_langevin_eq():
    sim = Langevin()
    sim.langevin_eq()
    sim.plot_x()
    sim.plot_z()
    sim.plot_spectrums()


def test_langevin_averaged():
    sim = langav.Langevin_averaged(*langav.initial_setup())
    sim.langevin_eq()
    sim.plot_xy_position()
    sim.plot("x")
    sim.plot("y")
    sim.plot("z")
    sim.plot_spectrums()
    sim.plot_summed_spectrum()


def test_oam_langevin():
    sim = oamlan.oam_Langevin(*inset.oam_trapping_initial_setup())
    sim.langevin_eq()
    location = "/Users/hnam/pycharm_projects/langevin/data"
    folder_name = "data"
    group_name = "test"
    sim.save_data(location, folder_name, group_name)
    # sim.plot_xy_position()
    # sim.plot("x")
    # sim.plot("y")
