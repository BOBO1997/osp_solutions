import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from pprint import pprint
import pickle
import time
import datetime

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler.passes import RemoveBarriers
import qiskit.quantum_info as qi

from osp_solutions.circuits_1d_path import gate_path_conventional, gate_path_triangle_parallel
from osp_solutions.backend_simulator import compute_distance_trace_unitary_mod_phase
from osp_solutions.hamiltonian import make_H_Heisenberg_path


filename_self = os.path.basename(__file__).split(".")[0]


### ================================================== ###


num_qubits = 9
time_evolution = 2 * np.pi
nums_steps_conventional = np.array(list(range(40, 400 + 1, 20)))
nums_steps_triangle = np.array(list(range(80, 800 + 1, 40)))
print("nums_steps_conventional: ", nums_steps_conventional)
print("nums_steps_triangle: ", nums_steps_triangle)

H_Heisenberg = make_H_Heisenberg_path(num_qubits=num_qubits) ### create Heisenberg Hamiltonian in a dictionary format
H = qi.SparsePauliOp.from_list(list(H_Heisenberg.items()))
print(H)
dm_H = H.to_matrix() ### same, np.allclose(dm_H, matrix_Heisenberg)
U = sp.linalg.expm(-1j * dm_H * time_evolution) ###! same, np.allclose(U, U_Heisenberg)


### ================================================== ###


fidelities_conventional = []
distances_trace_conventional = []

for num_steps in nums_steps_conventional:
    print("trotter steps: ", num_steps)
    t1 = time.perf_counter()
    matrix_path_qiskit = \
        qi.Operator(gate_path_conventional(num_qubits=num_qubits,
                                        num_steps=num_steps,
                                        dt=time_evolution / num_steps,
                                        type_block="3cnot",
                                        to_instruction=False,
                                        add_barrier=False,
                                       )
                   )# .data
    fidelities_conventional.append(qi.process_fidelity(matrix_path_qiskit, qi.Operator(U)))
    distances_trace_conventional.append(compute_distance_trace_unitary_mod_phase(matrix_path_qiskit.data, qi.Operator(U).data))

    t2 = time.perf_counter()
    print('process fidelity between conventional and U_Heisenberg = {:.4f}'.format(fidelities_conventional[-1]))
    print("time:", t2 - t1)
    print()


### ================================================== ###


fidelities_triangle = []
distances_trace_triangle = []

for num_steps in nums_steps_triangle:
    print("trotter steps: ", num_steps)
    t1 = time.perf_counter()
    matrix_path_triangle = \
        qi.Operator(gate_path_triangle_parallel(num_qubits=num_qubits,
                                                num_steps=num_steps // 4, ###!
                                                dt=2 * time_evolution / num_steps, ###!
                                                type_H_eff="new",
                                                to_instruction=False,
                                                add_barrier=False,
                                               )
                   )# .data
    fidelities_triangle.append(qi.process_fidelity(matrix_path_triangle, qi.Operator(U)))
    distances_trace_triangle.append(compute_distance_trace_unitary_mod_phase(matrix_path_triangle.data, qi.Operator(U).data))
    
    t2 = time.perf_counter()
    print('process fidelity between triangle and U_Heisenberg = {:.4f}'.format(fidelities_triangle[-1]))
    print("time:", t2 - t1)
    print()


### ================================================== ###


nums_cnots_conventional = []

for num_steps in nums_steps_conventional:
    
    print("trotter steps: ", num_steps)
    t1 = time.perf_counter()

    qc_conventional = QuantumCircuit(num_qubits)
    qc_conventional.compose(gate_path_conventional(num_qubits=num_qubits,
                                                  num_steps=num_steps,
                                                  dt=time_evolution / num_steps,
                                                  type_block="3cnot",
                                                  to_instruction=False,
                                                  add_barrier=False,
                                                 ),
                           qubits=list(range(num_qubits)),
                           inplace=True,)
    # qc_conventional = qc_conventional.assign_parameters({dt: time_evolution / num_steps})
    qc_conventional_t3 = transpile(RemoveBarriers()(qc_conventional), 
                               optimization_level=3, 
                               basis_gates=["sx", "cx", "rz"])
    nums_cnots_conventional.append(qc_conventional_t3.count_ops().get("cx"))

    t2 = time.perf_counter()
    print("number of CNOT gates in conventional:", nums_cnots_conventional[-1])
    print("time:", t2 - t1)
    print()


### ================================================== ###


nums_cnots_triangle = []

for num_steps in nums_steps_triangle:

    print("trotter steps: ", num_steps)
    t1 = time.perf_counter()

    qc_triangle = QuantumCircuit(num_qubits)
    qc_triangle.compose(gate_path_triangle_parallel(num_qubits=num_qubits,
                                                    num_steps=num_steps // 4, ###!
                                                    dt=2 * time_evolution / num_steps, ###!
                                                    type_H_eff="new",
                                                    to_instruction=False,
                                                    add_barrier=False,
                                                   ),
                       qubits=list(range(num_qubits)),
                       inplace=True,)
    # qc_triangle = qc_triangle.assign_parameters({dt: time_evolution / num_steps})
    qc_triangle_t3 = transpile(RemoveBarriers()(qc_triangle), 
                               optimization_level=3, 
                               basis_gates=["sx", "cx", "rz"])
    nums_cnots_triangle.append(qc_triangle_t3.count_ops().get("cx"))
    
    t2 = time.perf_counter()
    print("number of CNOT gates in triangle:", nums_cnots_triangle[-1])
    print("time:", t2 - t1)
    print()


### ================================================== ###


with open(filename_self+".pkl", "wb") as f:
    pickle.dump(obj={"nums_steps_conventional": np.array(nums_steps_conventional),
                     "fidelities_conventional": np.array(fidelities_conventional),
                     "distances_trace_conventional": np.array(distances_trace_conventional),
                     "nums_cnots_conventional": np.array(nums_cnots_conventional),
                     "nums_steps_triangle": np.array(nums_steps_triangle),
                     "fidelities_triangle": np.array(fidelities_triangle),
                     "distances_trace_triangle": np.array(distances_trace_triangle),
                     "nums_cnots_triangle": np.array(nums_cnots_triangle),
                    },
                file=f)
    

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