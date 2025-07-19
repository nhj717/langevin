"""
Run this file to run the beam profile code and save the data for plotting
"""

import initial_setup as inset
from oam_langevin import oam_Langevin
import time

start_time = time.time()
sim = oam_Langevin(*inset.oam_trapping_initial_setup())
(
    diameter,
    eps_glass,
    power,
    pressure,
    core_radius,
    N,
    delt,
    iteration,
    mode_number,
) = inset.oam_trapping_initial_setup()
sim.langevin_eq()
end_time = time.time()
print(f"simulation time was {end_time-start_time}")
location = "/Users/hnam/pycharm_projects/langevin/data"
folder_name = "langevin_data"
group_name = (
    f"l={mode_number}_{power}mW_{pressure}mbar_{round(N*delt,1)}s_{round(delt*1e6,1)}us"
)
print(f"data being saved as {group_name}")
sim.save_data(location, folder_name, group_name)
