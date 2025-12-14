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


fidelities_conventional = run_conventional["fidelities"]
nums_cnots_conventional = run_conventional["nums_cnots"]


fidelities_triangle_parallel = run_triangle_parallel["fidelities"]
nums_cnots_triangle_parallel = run_triangle_parallel["nums_cnots"]


plt.clf()
plt.figure(dpi=200)
p1 = plt.scatter(nums_steps_conventional, fidelities_conventional, marker="o")
plt.plot(nums_steps_conventional, fidelities_conventional,linewidth=1, linestyle='dotted')
p3 = plt.scatter(nums_steps_triangle / 2, fidelities_triangle_parallel, marker="x")
plt.plot(nums_steps_triangle / 2, fidelities_triangle_parallel,linewidth=1, linestyle='dashdot')
plt.xlabel(r"$n, \frac{m}{2}$")
plt.ylabel("state fidelity")
plt.ylim(bottom=-0.04,top=1.04)
plt.legend([p1, 
            p3],
           ["conventional",
            "proposed"])
plt.savefig(fname=filename_self+"_step_to_fidelity.png")


plt.clf()
plt.figure(dpi=200)
p1 = plt.scatter(nums_steps_conventional, 1 - np.array(fidelities_conventional), marker="o")
plt.plot(nums_steps_conventional, 1 - np.array(fidelities_conventional),linewidth=1, linestyle='dotted')
p3 = plt.scatter(nums_steps_triangle / 2, 1 - np.array(fidelities_triangle_parallel), marker="x")
plt.plot(nums_steps_triangle / 2, 1 - np.array(fidelities_triangle_parallel),linewidth=1, linestyle='dashdot')
plt.xlabel(r"$n, \frac{m}{2}$")
plt.ylabel("state infidelity")
# plt.grid(visible=True,which="minor")
plt.yscale("log")
# plt.yticks([1e-1,3*1e-1,4*1e-1,6*1e-1,8*1e-1,1e-0],
#            [r"$10^{-1}$",r"$2\times10^{-1}$",r"$4\times10^{-1}$",r"$6\times10^{-1}$",r"$8\times10^{-1}$",r"$10^{0}$"],
#            minor=True)
plt.legend([p1,  
            p3],
           ["conventional",
            "proposed"])
plt.savefig(fname=filename_self+"_step_to_infidelity_log.png")

plt.clf()
plt.figure(dpi=200)
p1 = plt.scatter(nums_steps_conventional, 1 - np.array(fidelities_conventional), marker="o")
plt.plot(nums_steps_conventional, 1 - np.array(fidelities_conventional),linewidth=1, linestyle='dotted')
p3 = plt.scatter(nums_steps_triangle / 2 * 7/6, 1 - np.array(fidelities_triangle_parallel), marker="x")
plt.plot(nums_steps_triangle / 2 * 7/6, 1 - np.array(fidelities_triangle_parallel),linewidth=1, linestyle='dashdot')
plt.xlabel(r"$n, \frac{m}{2}$")
plt.ylabel("state infidelity")
# plt.grid(visible=True,which="minor")
plt.yscale("log")
# plt.yticks([1e-1,3*1e-1,4*1e-1,6*1e-1,8*1e-1,1e-0],
#            [r"$10^{-1}$",r"$2\times10^{-1}$",r"$4\times10^{-1}$",r"$6\times10^{-1}$",r"$8\times10^{-1}$",r"$10^{0}$"],
#            minor=True)
plt.legend([p1,  
            p3],
           ["conventional",
            "proposed"])
plt.savefig(fname=filename_self+"_step_to_infidelity_log_7_over_6.png")


plt.clf()
plt.figure(dpi=200)
p1 = plt.scatter(nums_steps_conventional, nums_cnots_conventional, marker="o")
plt.plot(nums_steps_conventional, nums_cnots_conventional,linewidth=1, linestyle='dotted')
p3 = plt.scatter(nums_steps_triangle / 2, nums_cnots_triangle_parallel, marker="x")
plt.plot(nums_steps_triangle / 2, nums_cnots_triangle_parallel,linewidth=1, linestyle='dashdot')
plt.xlabel(r"$n, \frac{m}{2}$")
plt.ylabel("number of CNOT gates")
plt.legend([p1,
            p3],
            ["conventional",
             "proposed"])
plt.savefig(fname=filename_self+"_step_to_cnot.png")


plt.clf()
plt.figure(dpi=200)
p1 = plt.scatter(nums_cnots_conventional, fidelities_conventional, marker="o")
plt.plot(nums_cnots_conventional, fidelities_conventional, linewidth=1, linestyle='dotted')
p3 = plt.scatter(nums_cnots_triangle_parallel, fidelities_triangle_parallel, marker="x")
plt.plot(nums_cnots_triangle_parallel, fidelities_triangle_parallel, linewidth=1, linestyle='dashdot')
plt.xlabel("number of CNOT gates")
plt.ylabel("state fidelity")
plt.legend([p1, 
            p3], 
           ["conventional",  
            "proposed"])
plt.savefig(fname=filename_self+"_cnot_to_fidelity.png")


plt.clf()
plt.figure(dpi=200)
p1 = plt.scatter(nums_cnots_conventional, 1 - np.array(fidelities_conventional), marker="o")
plt.plot(nums_cnots_conventional, 1 - np.array(fidelities_conventional),linewidth=1, linestyle='dotted')
p3 = plt.scatter(nums_cnots_triangle_parallel, 1 - np.array(fidelities_triangle_parallel), marker="x")
plt.plot(nums_cnots_triangle_parallel, 1 - np.array(fidelities_triangle_parallel),linewidth=1, linestyle='dashdot')
plt.xlabel("number of CNOT gates")
plt.ylabel("state infidelity")
# plt.grid(visible=True,which="minor")
plt.yscale("log")
# plt.yticks([1e-1,3*1e-1,4*1e-1,6*1e-1,8*1e-1,1e-0],
#            [r"$10^{-1}$",r"$2\times10^{-1}$",r"$4\times10^{-1}$",r"$6\times10^{-1}$",r"$8\times10^{-1}$",r"$10^{0}$"],
#            minor=True)
plt.legend([p1, 
            p3], 
           ["conventional", 
            "proposed"])
plt.savefig(fname=filename_self+"_cnot_to_infidelity_log.png")