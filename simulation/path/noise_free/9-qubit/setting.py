import numpy as np

p_dep1 = 1.0 * 1e-4
p_dep2 = 1.0 * 1e-3

num_qubits = 9

# time_evolution = np.pi / 3
time_evolution = np.pi
# time_evolution = 2 * np.pi
# time_evolution = 3 * np.pi

initial_layout = list(range(num_qubits))

num_steps_list = list(range(20,500,40))

state_initial = "101010101" ###!
# state_initial = "110100110" ###!

type_H_eff_hybrid = "old"
type_H_eff_triangle = "new"

### ====== noise model ====== ###

from osp_solutions.backend_simulator import make_simulator_deplarising, simulator_ideal

simulator_ideal = simulator_ideal
simulator_noisy = make_simulator_deplarising(p_dep1=p_dep1,
                                             p_dep2=p_dep2)