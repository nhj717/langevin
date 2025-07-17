from fundamental_langevin import *

sim = Langevin_averaged(100)
sim.langevin_eq()
sim.plot_x()
sim.plot_z()
sim.plot_spectrums()
