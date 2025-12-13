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

from kagome_trotter.circuits_initial import gate_initial_state
from kagome_trotter.circuits_1d_path import gate_path_triangle_parallel
from kagome_trotter.hamiltonian import Hamiltonian, make_H_Heisenberg_path
from kagome_trotter.backend_simulator import result_to_dms, make_dm_binary, hamiltonian_to_dm, DMExtended
from setting import *
from kagome_trotter.backend_simulator import make_simulator_deplarising


filename_self = os.path.basename(__file__).split(".")[0]


dt = Parameter(r"$\Delta t$")
state_initial_dm = DMExtended(matrix=make_dm_binary(str_binary=state_initial,
                                                    endian_binary="big",
                                                    endian_dm="little")) ### make the density matrix for the initial state

H_Heisenberg = make_H_Heisenberg_path(num_qubits=num_qubits) ### create Heisenberg Hamiltonian in a dictionary format
matrix_Heisenberg = hamiltonian_to_dm(hamiltonian=H_Heisenberg) ### convert Heisenberg Hamiltonian to its matrix form

U_Heisenberg = DMExtended(sp.linalg.expm(-1.0j * matrix_Heisenberg * time_evolution)) ### prepare the unitary matrix of the Heisenberg Hamiltonian

state_target_dm = state_initial_dm.apply_unitary(U_Heisenberg) ### apply the unitary evolution to the density matrix
state_target_dm.is_valid()

fidelities_noise_level = []
nums_cnots_noise_level = []
for p_dep1, p_dep2 in zip(ps_dep1, ps_dep2):

    print("p_dep1:", p_dep1)
    print("p_dep2:", p_dep2)
    simulator_noisy = make_simulator_deplarising(p_dep1=p_dep1,
                                                 p_dep2=p_dep2)
    print(simulator_noisy)

    fidelities = []
    nums_cnots = []

    print("trotter step list: ", nums_steps_triangle)
    for num_steps in nums_steps_triangle:
        
        print("trotter steps: ", num_steps)
        t1 = time.perf_counter()
        
        # Initialize quantum circuit for 3 qubits
        qc = QuantumCircuit(num_qubits)

        # Prepare initial state
        qc.compose(gate_initial_state(state_initial=state_initial),
                qubits=list(range(num_qubits)),
                inplace=True,)
        
        qc.append(instruction=gate_path_triangle_parallel(num_qubits=num_qubits,
                                                        num_steps=num_steps // 4, ###!
                                                        dt=2 * dt, ###!
                                                        type_H_eff=type_H_eff_triangle,
                                                        to_instruction=False,
                                                        add_barrier=False,
                                                        ),
                qargs=list(range(num_qubits)),
                )

        # Evaluate simulation at target_time (t=pi) meaning each trotter step evolves pi/trotter_steps in time
        qc = qc.assign_parameters({dt: time_evolution / num_steps})
        qc_t3 = transpile(RemoveBarriers()(qc), optimization_level=3, basis_gates=["sx", "cx", "rz"])
        qc_t3.save_density_matrix()
        
        ### execute circuits
        result_raw = simulator_noisy.run(qc_t3,
                                        shots=0,
                                        ).result()
        dm_raw = result_to_dms(result=result_raw,
                            endian_result="little",
                            endian_dm="little",
                            )[0]

        fidelities.append(qi.state_fidelity(dm_raw, state_target_dm))
        nums_cnots.append(qc_t3.count_ops().get("cx"))
        
        t2 = time.perf_counter()
        print("raw state tomography fidelity = {:.4f}".format(fidelities[-1]))
        print("number of CNOT gates:", qc_t3.count_ops().get("cx"))
        print("time:", t2 - t1)
        print()

    fidelities_noise_level.append(fidelities)
    nums_cnots_noise_level.append(nums_cnots)


with open("run_triangle_parallel.pkl", "wb") as f:
    pickle.dump(obj={"fidelities_noise_level": fidelities_noise_level,
                     "nums_cnots_noise_level": nums_cnots_noise_level},
                file=f)