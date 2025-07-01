from langevin_averaged import *

sim = Langevin_averaged(10)
sim.langevin_eq()
sim.plot_x()
sim.plot_z()
sim.plot_spectrums()
