import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pprint import pprint
import pickle
import time
import datetime

# Import qubit states Zero (|0>) and One (|1>), and Pauli operators (X, Y, Z)
from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter
# from qiskit.tools.monitor import job_monitor
from qiskit.compiler import transpile
from qiskit.transpiler.passes import RemoveBarriers

# Import state tomography modules
from qiskit_experiments.library import StateTomography
from osp_solutions.patch_ignis import rename_qcs_StateTomography_to_ignis
from osp_solutions.tomography import StateTomographyFitter
from qiskit.quantum_info import state_fidelity

from osp_solutions.circuits_util import append_block_trotter_proposed, append_initial_state, append_block_trotter_aj, append_block_trotter_negishi
from osp_solutions.simulator_dm import make_dm_binary, hamiltonian_to_dm, DMExtended
from osp_solutions.hamiltonian import Hamiltonian, make_H_Heisenberg
from osp_solutions.backend_simulator import result_to_dms
from setting import *

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error
# from qiskit_ibm_runtime.fake_provider import FakeJakartaV2
# backend = AerSimulator.from_backend(FakeJakartaV2())
# backend = Aer.get_backend("qasm_simulator")

simulator_ideal = AerSimulator(method="density_matrix")

###? num_qubits = 5

# The final time of the state evolution
###? time_evolution = np.pi

# Parameterize variable t to be evaluated at t=pi later
dt = Parameter('t')

# initial layout
# initial_layout = [5,3,1]
###? initial_layout = list(range(num_qubits))

# Number of trotter steps
# num_steps_list = [4,10,50,100,200] # ,20,30,40,50] # ,60,70,80,90,100]
###? num_steps_list = list(range(4,100,4))
print("trotter step list: ", num_steps_list)

###? lmd = 1.0

###? state_initial_str = "10101" ### specify initial state
state_initial_dm = DMExtended(matrix=make_dm_binary(str_binary=state_initial_str, 
                                                    endian_binary="little_endian", 
                                                    endian_dm="big_endian")) ### make the density matrix for the initial state

H_Heisenberg = make_H_Heisenberg(num_qubits=num_qubits) ### create Heisenberg Hamiltonian in a dictionary format
matrix_Heisenberg = hamiltonian_to_dm(hamiltonian=H_Heisenberg) ### convert Heisenberg Hamiltonian to its matrix form

U_Heisenberg = DMExtended(sp.linalg.expm(-1.0j * matrix_Heisenberg * time_evolution)) ### prepare the unitary matrix of the Heisenberg Hamiltonian

state_target_dm = state_initial_dm.apply_unitary(U_Heisenberg) ### apply the unitary evolution to the density matrix
state_target_dm.is_valid()


noiselevels_to_fidelities = []
noiselevels_to_nums_cnots = []

for p_dep1, p_dep2 in zip(ps_dep1, ps_dep2):
    ###? p_dep1 = 1.0 * 1e-4
    error_dep1 = pauli_error([("I", 1 - 3 * p_dep1 / 4), ("X", p_dep1 / 4), ("Y", p_dep1 / 4), ("Z", p_dep1 / 4)])
    # error_dep2_local = error_dep1.tensor(error_dep1)

    ###? p_dep2 = 1.0 * 1e-3
    error_dep2_global = pauli_error([("II", 1 - 15 * p_dep2 / 16), ("IX", p_dep2 / 16), ("IY", p_dep2 / 16), ("IZ", p_dep2 / 16),
                                    ("XI", p_dep2 / 16), ("XX", p_dep2 / 16), ("XY", p_dep2 / 16), ("XZ", p_dep2 / 16),
                                    ("YI", p_dep2 / 16), ("YX", p_dep2 / 16), ("YY", p_dep2 / 16), ("YZ", p_dep2 / 16),
                                    ("ZI", p_dep2 / 16), ("ZX", p_dep2 / 16), ("ZY", p_dep2 / 16), ("ZZ", p_dep2 / 16)])

    # print(error_dep2_global)

    # Add errors to noise model
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_dep1, ["rx", "rz", "sx", "h", "sdg", "s", "x", "u1", "u2", "u3"])
    noise_model.add_all_qubit_quantum_error(error_dep2_global, ["cx", "cz"])
    # noise_model.add_all_qubit_quantum_error(error_3, ["cswap", "ccx"])
    print(noise_model)
    print()

    # Create noisy simulator backend
    simulator_noisy = AerSimulator(method="density_matrix",
                                noise_model=noise_model)
    
    fidelities = []
    nums_cnots = []

    for num_steps in num_steps_list:
        
        print("trotter steps: ", num_steps)
        t1 = time.perf_counter()
        
        # Initialize quantum circuit for 3 qubits
        qc = QuantumCircuit(num_qubits)

        # Prepare initial state
        append_initial_state(qc=qc, 
                             state_initial_str=state_initial_str)
        append_block_trotter_proposed(qc=qc, 
                                      dt=dt,  ###!
                                      lmd= lmd, ### !
                                      num_steps=num_steps, ###!
                                     )

        # Evaluate simulation at target_time (t=pi) meaning each trotter step evolves pi/trotter_steps in time
        qc = qc.assign_parameters({dt: time_evolution / num_steps})
        qc_t3 = transpile(RemoveBarriers()(qc), 
                          optimization_level=3, 
                          basis_gates=["sx", "cx", "rz"])
        qc_t3.save_density_matrix()
        
        ### execute circuits
        result_raw = simulator_noisy.run(qc_t3,
                                            shots=0,
                                            ).result()
        dm_raw = result_to_dms(result=result_raw,
                            endian_dm="big")[0]

        fidelities.append(state_fidelity(dm_raw, state_target_dm))
        nums_cnots.append(qc_t3.count_ops().get("cx"))
        
        t2 = time.perf_counter()
        print('raw state tomography fidelity = {:.4f} \u00B1 {:.4f}'.format(np.mean(fidelities), np.std(fidelities)))
        print("number of CNOT gates:", qc_t3.count_ops().get("cx"))
        print("time:", t2 - t1)
        print()
    
    noiselevels_to_fidelities.append(fidelities)
    noiselevels_to_nums_cnots.append(nums_cnots)

with open("run_proposed.pkl", "wb") as f:
    pickle.dump(obj={"noiselevels_to_fidelities": noiselevels_to_fidelities, 
                     "noiselevels_to_nums_cnots": noiselevels_to_nums_cnots}, 
                file=f,
               )