import numpy as np

p_dep1 = 0 # 1.0 * 1e-4
p_dep2 = 0 # 1.0 * 1e-3

num_qubits = 9

times_evolution_continuous = np.linspace(0, np.pi, 100)
times_evolution_discrete = [i * np.pi / 20 for i in range(0, 20 + 1)]

initial_layout = list(range(num_qubits))

# num_steps_list = list(range(4,100,4))
num_steps = 30

state_initial_str = "110100110" # "101010101" ### specify initial state, 10101, 11100

lmd = 1.0