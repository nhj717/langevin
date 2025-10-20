"""
Plot from the saved data of the beam profiles
"""

import shared_function
from plot_save import *

mode_number = 0
power = 150
pressure = 2
delt = 1e-6
N = 1e6
location = "/Users/hnam/pycharm_projects/langevin/data"
file_name = "langevin_data"
group_name = (
    f"l={mode_number}_{power}mW_{pressure}mbar_{round(N*delt,1)}s_{round(delt*1e6,1)}us"
)
print(f"reading from {group_name}")
