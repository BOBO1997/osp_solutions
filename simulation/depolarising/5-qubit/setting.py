import numpy as np

p_dep1 = 1.0 * 1e-4
p_dep2 = 1.0 * 1e-3

num_qubits = 5

time_evolution = np.pi

initial_layout = list(range(num_qubits))

num_steps_list = list(range(4,100,4))

state_initial_str = "11010" ### specify initial state

lmd = 1.0