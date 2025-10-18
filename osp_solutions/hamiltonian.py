from typing import *
from collections.abc import Mapping
import copy, time
import numpy as np
import qiskit.quantum_info as qi

import networkx as nx
from osp_solutions.simulator_dm import hamiltonian_to_dm

class Hamiltonian(Mapping):
    def __init__(self, arg_dict: Dict[str, complex]):
        self.num_qubits = None if len(arg_dict) == 0 else len(list(arg_dict.keys())[0])
        self.operators = arg_dict
        self.pindex = {"I": 0, "X": 1, "Y": 2, "Z": 3}
        self.ptable = [[("I", 1.0), ("X", 1.0), ("Y", 1.0), ("Z", 1.0)],
                       [("X", 1.0), ("I", 1.0), ("Z", 1.j), ("Y", -1.j)],
                       [("Y", 1.0), ("Z", -1.j), ("I", 1.0), ("X", 1.j)],
                       [("Z", 1.0), ("Y", 1.j), ("X", -1.j), ("I", 1.0)]]

    def __add__(self, arg_dict):
        ret = copy.deepcopy(self.operators)
        for operator, coeff in arg_dict.items():
            if operator in ret:
                ret[operator] += coeff
            else:
                ret[operator] = coeff
            if ret[operator] == 0:
                del ret[operator]
        return Hamiltonian(ret)

    def __sub__(self, arg_dict):
        ret = copy.deepcopy(self.operators)
        for operator, coeff in arg_dict.items():
            if operator in ret:
                ret[operator] -= coeff
            else:
                ret[operator] = - coeff
            if ret[operator] == 0:
                del ret[operator]
        return Hamiltonian(ret)

    def __mul__(self, arg_dict):
        ret = dict()
        for left_operator, left_coeff in self.operators.items():
            for right_operator, right_coeff in arg_dict.items():
                temp_operator = ""
                temp_coeff = left_coeff * right_coeff
                for left_pauli, right_pauli in zip(left_operator, right_operator):
                    new_pauli, new_coeff = self.ptable[self.pindex[left_pauli]][self.pindex[right_pauli]]
                    temp_operator += new_pauli
                    temp_coeff *= new_coeff
                if temp_operator in ret:
                    ret[temp_operator] += temp_coeff
                else:
                    ret[temp_operator] = temp_coeff
                if ret[temp_operator] == 0:
                    del ret[temp_operator]
        return Hamiltonian(ret)

    def __pow__(self, num_power):
        if len(self.operators) == 0:
            raise Exception("Undefined behaviour.")
        ret = Hamiltonian({"I"*len(next(iter(self.operators))): 1.0})
        H_pow_temp = Hamiltonian(self)
        for i, digit in enumerate(format(num_power, "0b")[::-1]):
            if i > 0:
                H_pow_temp *= H_pow_temp
            if digit == "1":
                ret *= H_pow_temp
        return ret

    def __setitem__(self, key, value):
        self.operators[key] = value

    def __getitem__(self, key):
        return self.operators.get(key, 0.0)

    def __iter__(self):
        return iter(self.operators)

    def __len__(self):
        return len(self.operators)

    def __repr__(self):
        return repr(self.operators)

    def __str__(self):
        return str(self.operators)

    def __hash__(self):
        return hash(str(sorted(self.operators)))

    def __eq__(self, other):
        return hash(self) == hash(other)
    
    def to_json(self):
        return str(self)

    def reduce_identity(self, inplace=False):
        if inplace:
            del self.operators["I" * self.num_qubits]
        else:
            ret = Hamiltonian({})
            for pauli_str, coeff in self.items():
                if not pauli_str == "I" * len(pauli_str):
                    ret[pauli_str] = coeff
            return ret
    
    def compute_energy_of_ground_state(self, size_limit = 8):
        if self.num_qubits > size_limit:
            raise Exception("exceed the size limit.")
        matrix_H = np.zeros([self.num_qubits, self.num_qubits], dtype="complex")
        for operator, coeff in self.operators:
            matrix_H += coeff * qi.Pauli(operator).to_matrix()
        energies_theoretical, _ = np.linalg.eig(matrix_H)
        energy_theoretical = sorted(energies_theoretical.real)[0]
        return energy_theoretical


def extract_sub_pauli_str(pauli_str: str, 
                          poses: List[int]) -> str:
    ret = ""
    for pos in poses:
        ret += pauli_str[pos]
    return ret


def divide_pauli_str(pauli_str: str, 
                     section_list: List[List[int]]) -> List[str]:
    ret = []
    for section in section_list:
        ret.append(pauli_str, section)
    return ret


def divide_pauli_str_equally(pauli_str: str, 
                             num_divide: int) -> List[str]:

    # Calculate the length of each part
    part_length = len(pauli_str) // num_divide

    # Use list comprehension to split the string
    parts = [pauli_str[i:i + part_length] for i in range(0, len(pauli_str), part_length)]

    return parts


def extract_sub_hamiltonian(hamiltonian: Hamiltonian, 
                            poses: List[int], 
                            reduce_identity=False) -> Hamiltonian:
    sub_hamiltonian = Hamiltonian({})
    for pauli_str, coeff in hamiltonian.items():
        sub_hamiltonian += Hamiltonian({extract_sub_pauli_str(pauli_str, poses): coeff})
    if reduce_identity:
        sub_hamiltonian.reduce_identity(inplace=True)
    return sub_hamiltonian


def divide_hamiltonian(hamiltonian: Hamiltonian, 
                       section_list: List[List[int]], 
                       reduce_identity=False) -> List[Hamiltonian]:
    sub_hamiltonians = [Hamiltonian({}) for _ in section_list]
    for pauli_str, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        for m, section in enumerate(section_list):
            sub_hamiltonians[m] += Hamiltonian({extract_sub_pauli_str(pauli_str, section): coeff})
    if reduce_identity:
        for sub_hamiltonian in sub_hamiltonians:
            sub_hamiltonian.reduce_identity(inplace=True)
    return sub_hamiltonians


def make_powers_H(hamiltonian: Hamiltonian,
                  power_max: int):
    powers_H = dict()
    for i in range(power_max):
        t1 = time.perf_counter()
        if i == 0:
            powers_H[i] = hamiltonian ** 0
        else:
            powers_H[i] = powers_H[i - 1] * hamiltonian
        t2 = time.perf_counter()
        print(i, "th power of H finished, time", t2 - t1, "s, number of terms in Hamiltonian:", len(powers_H[i]))
    return powers_H


### functions to make each Hamiltonian for each physical model ###
### set Hamiltonian (spanning whole system!) ###
def make_H_ghz(size_system: int):
    H = Hamiltonian({})
    observable = "X" * size_system
    H += {observable: -1.0}
    for i in range(size_system):
        if i < size_system - 1:
            observable = list("I" * size_system)
            observable[i] = observable[i + 1] = "Z"
            observable = "".join(observable)
            H += {observable: -1.0}
    return H

### set Hamiltonian used in the subspace ###
def make_H_subspace_ghz(H: Hamiltonian,
                        num_divisions: int):
    num_qubits_qreg_dgse = len(list(H.keys())[0]) // num_divisions
    H_subspace = Hamiltonian({})
    for str_pauli, coeff in H.items():
        for pos_division in range(1, num_divisions):
            if (not (str_pauli[pos_division * num_qubits_qreg_dgse - 1] == str_pauli[pos_division * num_qubits_qreg_dgse] == "I")) and (str_pauli not in H_subspace.operators): ### for the whole tomography ###
            # if (str_pauli[pos_division * num_qubits_qreg_dgse - 1] != "I") and (str_pauli[pos_division * num_qubits_qreg_dgse] != "I") and (str_pauli not in H_subspace.operators): ### for the lightest tomography ###
                H_subspace += {str_pauli: coeff} ### actually the coefficient is not important ###
    # H_subspace = H ### * if we want to use the full power subspace * ###
    return H_subspace

def make_H_fidelity_ghz(generators_stabilizers: List[Hamiltonian],
                        num_qubits: int,
                        num_stabilizers: int,
                        seed_random: int = 42):
    H_fidelity = Hamiltonian({})
    for ith_sample in range(num_stabilizers):
        sample_randints = np.random.randint(2, size=num_qubits) ### TODO: specify random seed ###
        sample_stabilizer = Hamiltonian({"I" * num_qubits: (1.0 + 0.0j) / num_stabilizers})
        for ith_qubit, randint in enumerate(sample_randints):
            if randint == 1:
                sample_stabilizer *= generators_stabilizers[ith_qubit]
        assert np.allclose(list(sample_stabilizer.values())[0].imag, 0.0)
        H_fidelity += sample_stabilizer
    return H_fidelity

### set Hamiltonian of 1D Ising model ###
def make_H_Ising_1D(num_qubits: int) -> Tuple[Hamiltonian, float]:
    n_rows = 1
    n_cols = num_qubits
    num_qubits = n_rows * n_cols
    # print("num_qubits:", num_qubits)
    # print()
    h = 1 * np.ones((n_rows, n_cols)) # Set the value of the external magnetic field at each site.

    G = nx.Graph()
    for i in range(num_qubits - 1):
        G.add_edge(i, i + 1)

    ### set Hamiltonian ###
    H = Hamiltonian({})
    for i in range(num_qubits):
        observable = list("I" * num_qubits)
        observable[i] = "X"
        observable = "".join(observable)
        H += {observable: -h.flatten()[i]}
    for i in range(n_rows):
        for j in range(n_cols):
            if i < n_rows - 1:
                observable = list("I" * num_qubits)
                observable[i * n_cols + j] = observable[(i + 1) * n_cols + j] = "Z"
                observable = "".join(observable)
                H += {observable: -1}
            if j < n_cols - 1:
                observable = list("I" * num_qubits)
                observable[i * n_cols + j] = observable[i * n_cols + j + 1] = "Z"
                observable = "".join(observable)
                H += {observable: -1}
    # print("Hamiltonian:")
    # print(H)
    # print()
    energies_theoretical = np.linalg.eig(hamiltonian_to_dm(H))[0]
    energy_theoretical = sorted(energies_theoretical.real)[0]
    # print("theoretical ground state energy:", energy_theoretical)
    # print()
    # print("top four energies:", sorted(energies_theoretical.real)[:4])
    return H, energy_theoretical

def make_H_Heisenberg(num_qubits: int) -> Hamiltonian:
    H = Hamiltonian({})
    for ith_qubit in range(num_qubits - 1):
        observable = list("I" * num_qubits)
        observable[ith_qubit] = observable[ith_qubit + 1] = "X"
        observable = "".join(observable)
        H += {observable: 1}
        observable = list("I" * num_qubits)
        observable[ith_qubit] = observable[ith_qubit + 1] = "Y"
        observable = "".join(observable)
        H += {observable: 1}
        observable = list("I" * num_qubits)
        observable[ith_qubit] = observable[ith_qubit + 1] = "Z"
        observable = "".join(observable)
        H += {observable: 1}
    return H