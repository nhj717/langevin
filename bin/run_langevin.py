"""
Run this file to run the beam profile code and save the data for plotting
"""

import initial_setup as inset
from oam_langevin import oam_Langevin

sim = oam_Langevin(*inset.oam_trapping_initial_setup())
sim.langevin_eq()
location = "/Users/hnam/pycharm_projects/langevin/data"
folder_name = "langevin_data"
group_name = "l=1_400mW_1mbar_0.1s_1us"
sim.save_data(location, folder_name, group_name)
