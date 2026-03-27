import initial_setup as inset
from oam_langevin import oam_Langevin

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
print(sim.gamma0)
print(sim.m)

f_grad = 2.24 * 1e-16
v_eq = f_grad / sim.m / sim.gamma0
print(v_eq)
