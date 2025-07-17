"""
Plot from the saved data of the beam profiles
"""

from plot_save import *

location = "/Users/hnam/pycharm_projects/langevin/data"
file_name = "beam_profile_data"
group_name = "XY_data"
fig = plot_XY_with_Poynting(location, file_name, group_name)
# fig = plot_XZ(location, file_name, group_name)
image_name = "Intensity_XY"
save_figure(fig, image_name)
