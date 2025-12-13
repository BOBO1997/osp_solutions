import numpy as np

ps_dep1 = np.array([1.0 * 1e-7,
                    3.0 * 1e-7,
                    1.0 * 1e-6,
                    3.0 * 1e-6,
                    1.0 * 1e-5,
                   ])
ps_dep2 = ps_dep1 * 10

num_qubits = 9

time_evolution = np.pi

nums_steps_conventional = np.array(list(range(40, 400 + 1, 20)))
nums_steps_triangle = np.array(list(range(80, 800 + 1, 40)))

state_initial = "101010101" ###!

type_H_eff_triangle = "new"