import numpy as np

ps_dep1 = [0.0, 1.0 * 1e-6, 3.0 * 1e-6, 1.0 * 1e-5, 3.0 * 1e-5, 1.0 * 1e-4, 3.0 * 1e-4,]
ps_dep2 = [0.0, 1.0 * 1e-5, 3.0 * 1e-5, 1.0 * 1e-4, 3.0 * 1e-4, 1.0 * 1e-3, 3.0 * 1e-3,]

num_qubits = 9

time_evolution = np.pi

initial_layout = list(range(num_qubits))

num_steps_list = list(range(4,100,4))

state_initial_str = "110100110" # "101010101" ### specify initial state, 10101, 11100

lmd = 1.0