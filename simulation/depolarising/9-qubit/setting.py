import numpy as np

p_dep1 = 1.0 * 1e-4
p_dep2 = 1.0 * 1e-3

num_qubits = 9

time_evolution = np.pi

initial_layout = list(range(num_qubits))

num_steps_list = list(range(4,300,12))

state_initial_str = "110100110" # "101010101" ### specify initial state, 10101, 11100

lmd = 1.0