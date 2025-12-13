import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from pprint import pprint
import pickle
import time

filename_self = os.path.basename(__file__).split(".")[0]


### ================================================== ###


with open("fidelity_process_triangle_parallel_exact.pkl", "rb") as f:
    fidelity_process_triangle_parallel = pickle.load(f)
    for key, value in fidelity_process_triangle_parallel.items():
        if key.split("_")[-1] == "conventional":
            print(key)
            globals()[key] = value
        else:
            print(key, value)
print()

with open("fidelity_process_conventional.pkl", "rb") as f:
    fidelity_process_conventional = pickle.load(f)
    for key, value in fidelity_process_conventional.items():
        print(key, value)
        globals()[key] = value
    

### ================================================== ###


plt.clf()
plt.plot(nums_steps_conventional, 1 - np.array(fidelities_conventional), linestyle="dashed")
plt.plot(nums_steps_triangle / 2, 1 - np.array(fidelities_triangle), linestyle="dotted")
plt.xlabel(r"$n, \frac{m}{2}$")
plt.ylabel("process infidelity")
plt.savefig(fname=filename_self+"_step_to_infidelity.png")


plt.clf()
p3, = plt.plot(nums_steps_conventional, 1 - np.array(fidelities_conventional), linestyle="dashed")
p4, = plt.plot(nums_steps_triangle / 2, 1 - np.array(fidelities_triangle), linestyle="dotted")
plt.xlabel(r"$n, \frac{m}{2}$")
plt.ylabel("process infidelity")
plt.yscale("log")
plt.legend([p3,p4],["conventional","triangle"])
plt.savefig(fname=filename_self+"_step_to_infidelity_log.png")


plt.clf()
plt.plot(nums_steps_conventional, distances_trace_conventional, linestyle="dashed")
plt.plot(nums_steps_triangle / 2, distances_trace_triangle, linestyle="dotted")
plt.xlabel(r"$n, \frac{m}{2}$")
plt.ylabel("trace distance")
plt.savefig(fname=filename_self+"_step_to_distance.png")


plt.clf()
p3, = plt.plot(nums_steps_conventional, distances_trace_conventional, linestyle="dashed")
p4, = plt.plot(nums_steps_triangle / 2, distances_trace_triangle, linestyle="dotted")
plt.xlabel(r"$n, \frac{m}{2}$")
plt.ylabel("trace distance")
plt.yscale("log")
plt.legend([p3,p4],["conventional","triangle"])
plt.savefig(fname=filename_self+"_step_to_distance_log.png")


plt.clf()
p3, = plt.plot(nums_cnots_conventional, 1 - np.array(fidelities_conventional), linestyle="dashed")
p4, = plt.plot(nums_cnots_triangle, 1 - np.array(fidelities_triangle), linestyle="dotted")
plt.xlabel("number of CNOT gates")
plt.ylabel("process infidelity")
plt.yscale("log")
plt.legend([p3,p4],["conventional","triangle"])
plt.savefig(fname=filename_self+"_cnot_to_infidelity.png")