"""
Plot from the saved data of the beam profiles
"""

from plot_save import *

location = "/Users/hnam/pycharm_projects/langevin/data"
file_name = "beam_profile_data"
group_name = "XY_data"
fig = plot_1D(location, file_name, group_name)
# fig.savefig(
#     "/Users/hnam/ownCloud/phd stuff/poster/evaluation/xy_view2.png", pad_inches=0.02
# )
# fig = plot_XZ(location, file_name, group_name)
# image_name = "Intensity_XY"
# save_figure(fig, image_name)
