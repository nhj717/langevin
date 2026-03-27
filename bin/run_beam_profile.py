"""
Run this file to run the beam profile code and save the data for plotting
"""

import initial_setup as inset
from beam_profile import OAM_profile

sim = OAM_profile(*inset.beam_profile_initial_setup_XY(1, 0, [1, 0], [1, 0]))
# sim.S_and_I()
sim.standing_S_and_I_mix()
location = "/Users/hnam/pycharm_projects/langevin/data"
folder_name = "beam_profile_data"
group_name = "XY_data"
sim.save_data(location, folder_name, group_name)
