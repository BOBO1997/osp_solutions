import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from pprint import pprint
import pickle
from setting import *

filename_self = os.path.basename(__file__).split(".")[0]


with open("run_conventional.pkl", "rb") as f:
    run_conventional = pickle.load(f)
with open("run_triangle_parallel.pkl", "rb") as f:
    run_triangle_parallel = pickle.load(f)


fidelities_noise_level_conventional = run_conventional["fidelities_noise_level"]
nums_cnots_noise_level_conventional = run_conventional["nums_cnots_noise_level"]
fidelities_max_conventional = np.max(fidelities_noise_level_conventional, axis=1)

fidelities_noise_level_triangle_parallel = run_triangle_parallel["fidelities_noise_level"]
nums_cnots_noise_level_triangle_parallel = run_triangle_parallel["nums_cnots_noise_level"]
fidelities_max_triangle_parallel = np.max(fidelities_noise_level_triangle_parallel, axis=1)


plt.clf()
plt.figure(dpi=200)
p1 = plt.scatter(ps_dep1, 1 - fidelities_max_conventional, marker="o")
plt.plot(ps_dep1, 1 - fidelities_max_conventional, linewidth=1, linestyle='dotted')
p2 = plt.scatter(ps_dep1, 1 - fidelities_max_triangle_parallel, marker="x")
plt.plot(ps_dep1, 1 - fidelities_max_triangle_parallel, linewidth=1, linestyle='dashdot')
plt.xlabel("single-qubit depolarising probability")
plt.ylabel("state infidelity")
plt.xscale("log")
plt.yscale("log")
plt.legend([p1, 
            p2],
           ["conventional",
            "proposed"])
plt.savefig(fname=filename_self+"_noise-level_to_infidelity.png")