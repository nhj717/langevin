"""
Plot from the saved data of the beam profiles
"""

import shared_function
from plot_save import *

mode_number = 0
power = 200
pressure = 1
delt = 1e-6
N = 1e5

location = "/Users/hnam/pycharm_projects/langevin/data"
file_name = "langevin_data"
group_name = (
    f"l={mode_number}_{power}mW_{pressure}mbar_{round(N*delt,1)}s_{round(delt*1e6,1)}us"
)
print(f"reading from {group_name}")
data_label, data = shared_function.read_data(location, file_name, group_name)
m, N, gamma0, t, f, x, v = (
    data[data_label.index("m")],
    data[data_label.index("N")],
    data[data_label.index("gamma0")],
    data[data_label.index("t")],
    data[data_label.index("f")],
    data[data_label.index("x")],
    data[data_label.index("v")],
)

fig1, fig2, fig3 = plot_package(m, N, t, f, x, v, "x")
image_name1, image_name2, image_name3 = "x_vs_t", "p_x_vs_x", "xfft_vs_f"

save_figure(fig1, image_name1)
save_figure(fig2, image_name2)
save_figure(fig3, image_name3)

fig4, fig5, fig6 = plot_package(m, N, t, f, x, v, "y")
image_name4, image_name5, image_name6 = "y_vs_t", "p_y_vs_y", "yfft_vs_f"

save_figure(fig4, image_name4)
save_figure(fig5, image_name5)
save_figure(fig6, image_name6)

fig7, fig8, fig9 = plot_package(m, N, t, f, x, v, "z")
image_name7, image_name8, image_name9 = "z_vs_t", "p_z_vs_z", "zfft_vs_f"

save_figure(fig7, image_name7)
save_figure(fig8, image_name8)
save_figure(fig9, image_name9)

fig = plot_particle_xy(x)
image_name = "xy_position"
save_figure(fig, image_name)

fig = plot_spectrums(N, gamma0, f, x)
image_name = "mixed_spectrum"
save_figure(fig, image_name)

fig = plot_summed_spectrum(N, f, x)
image_name = "summed_spectrum"
save_figure(fig, image_name)
