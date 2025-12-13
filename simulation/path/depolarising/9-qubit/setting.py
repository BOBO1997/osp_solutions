import numpy as np

p_dep1 = 1.0 * 1e-6
p_dep2 = 1.0 * 1e-5

num_qubits = 9

time_evolution = np.pi

initial_layout = list(range(num_qubits))

nums_steps_conventional = np.array(list(range(4, 200 + 1, 8)))
nums_steps_triangle = np.array(list(range(8, 400 + 1, 16)))

state_initial = "101010101" ###!
# state_initial = "110100110" ###!

type_H_eff_hybrid = "old"
type_H_eff_triangle = "new"

### ====== noise model ====== ###

from kagome_trotter.backend_simulator import make_simulator_deplarising, simulator_ideal

simulator_ideal = simulator_ideal
simulator_noisy = make_simulator_deplarising(p_dep1=p_dep1,
                                             p_dep2=p_dep2)