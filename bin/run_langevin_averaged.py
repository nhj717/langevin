from langevin_averaged import *

sim = Langevin_averaged(1000)
sim.langevin_eq()
sim.plot_x()
sim.plot_z()
sim.plot_spectrums()
