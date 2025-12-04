from typing import *
import numpy as np
import copy
from qiskit.circuit import QuantumCircuit
from qiskit_aer.primitives import SamplerV2 as AerSampler
import qiskit.quantum_info as qi
from qiskit.result import Counts, Result #! deprecated 


# --- Pauli matrices ---

matrix_I = np.array([[1, 0], 
                     [0, 1]], 
                    dtype=complex)
matrix_X = np.array([[0, 1], 
                     [1, 0]], 
                    dtype=complex)
matrix_Y = np.array([[0, -1j], 
                     [1j, 0]], 
                    dtype=complex)
matrix_Z = np.array([[1, 0], 
                     [0, -1]],
                    dtype=complex)
matrix_Zero = np.array([[1, 0], 
                        [0, 0]],
                       dtype=complex)
matrix_One = np.array([[0, 0], 
                       [0, 1]],
                      dtype=complex)
matrix_Plus = np.array([[1, 1], 
                        [1, 1]],
                       dtype=complex)
matrix_Minus = np.array([[1, -1], 
                         [-1, 1]],
                        dtype=complex)



class DMExtended(qi.DensityMatrix):
    def __init__(self, 
                 matrix, 
                 dims=None):
        super().__init__(matrix, dims=dims)

    def __matmul__(self, 
                   other):
        if isinstance(other, qi.DensityMatrix):
            return DMExtended(self._data @ other._data, dims=self.dims())
        else:
            return DMExtended(self._data @ other, dims=self.dims())
        
    def __pow__(self, 
                num_power: int):
        if num_power == 0:
            return DMExtended(np.eye(self.dim), dims=self.dims())
        if num_power == 1:
            return DMExtended(self._data, dims=self.dims())
        else:
            ret = DMExtended(np.eye(self.dim), dims=self.dims())
            dm_pow_temp = DMExtended(self._data, dims=self.dims())
            for i, digit in enumerate(format(num_power, "0b")[::-1]):
                if i > 0: # for the smallest digit, we do not have to multiply the density matrix
                    dm_pow_temp @= dm_pow_temp
                if digit == "1":
                    ret @= dm_pow_temp
            return ret
    
    def dagger(self):
        return DMExtended(self._data.T.conjugate(), dims=self.dims())
    
    def tensor_pow(self, 
                   k: int):
        if k <= 0:
            return 0
        elif k == 1:
            return DMExtended(self._data, dims=self.dims())
        else:
            return DMExtended(self._data, dims=self.dims()) ^ DMExtended(self._data, dims=self.dims()).tensor_pow(k - 1)

    def normalize(self):
        return DMExtended(self._data / self.trace(), dims=self.dims())
    
    def partial_trace(self, 
                      qubits: List[int], 
                      normalize: bool = True):
        """
        qubtis: the subsystem to be traced over
        [with normalization]
        """
        if normalize:
            return DMExtended(qi.partial_trace(self._data, qubits)).normalize()
        else:
            return DMExtended(qi.partial_trace(self._data, qubits))
    
    def apply_unitary(self, 
                      U: qi.DensityMatrix, 
                      normalize: bool = True):
        """
        [with normalization]
        """
        if not isinstance(U, qi.DensityMatrix):
            U = DMExtended(U)
        if normalize:
            return DMExtended(U @ self @ U.dagger()).normalize()
        else:
            return DMExtended(U @ self @ U.dagger())


def qc_to_dm(qc: QuantumCircuit,
             endian_qc: str = "big",
             endian_dm: str = "little",
            ) -> DMExtended:
    """
    Note that you have to add `qc.save_density_matrix()`.
    `result.data()["density_matrix"]` is equivalent to
    `result.results[i].data.density_matrix`

    big endian:    q_0 = X, q_1 = Y, q_2 = Z -> q_0 q_1 q_2 = "XYZ", i.e. X \otimes Y \otimes Z
    little endian: q_0 = X, q_1 = Y, q_2 = Z -> q_2 q_1 q_0 = "ZYX", i.e. Z \otimes Y \otimes X

    That is, if one runs `for i, state in enumerate(some_str)` and apply `some_str[0] \otimes some_str[1] \otimes ...`,
    this is big endian to big endian, and little endian to little endian.
    Note that the QuantumCircuit instance adopts the list of qubit indices (q_0, q_1, ...) in a big endian style.
    """
    ###! by default: endian_qc is big == endian_result is little == endian_dm is little
    flag_reverse_endian = -1 if endian_qc == endian_dm else 1

    sampler = AerSampler() ### here, backend is not used ###
    job = sampler.run(qc, shots=0)
    result = job.result()
    dm = result.data()["density_matrix"]
    if flag_reverse_endian:
        return DMExtended(dm.reverse_qargs())
    else:
        return DMExtended(dm)
    

def str_state_to_dm(str_state: str,
                    endian_str: str = "big",
                    endian_dm: str = "little",
                   ) -> DMExtended:
    """
    ###! maybe, using DensityMatrix.from_label is better
    ###! qi.DensityMatrix.from_label is from a label string with big endian to a density matrix with little endian

    big endian:    q_0 = X, q_1 = Y, q_2 = Z -> q_0 q_1 q_2 = "XYZ", i.e. X \otimes Y \otimes Z
    little endian: q_0 = X, q_1 = Y, q_2 = Z -> q_2 q_1 q_0 = "ZYX", i.e. Z \otimes Y \otimes X

    That is, if one runs `for i, state in enumerate(some_str)` and apply `some_str[0] \otimes some_str[1] \otimes ...`,
    this is big endian to big endian, and little endian to little endian.
    Note that the QuantumCircuit instance adopts the list of qubit indices (q_0, q_1, ...) in a big endian style.
    # """
    flag_reverse_endian = 1 if endian_str == endian_dm else -1
    dm = 1
    for ith_char in str_state[::flag_reverse_endian]:
        if ith_char == "0":
            dm = np.kron(dm, matrix_Zero)
        elif ith_char == "1":
            dm = np.kron(dm, matrix_One)
        else:
            raise Exception("please specify 0 or 1 only.")
    return DMExtended(dm)


def str_pauli_to_dm(str_pauli: Union[str, qi.Pauli],
                    endian_pauli: str = "big",
                    endian_dm: str = "little",
                   ) -> DMExtended:
    """
    big endian:    q_0 = X, q_1 = Y, q_2 = Z -> q_0 q_1 q_2 = "XYZ", i.e. X \otimes Y \otimes Z
    little endian: q_0 = X, q_1 = Y, q_2 = Z -> q_2 q_1 q_0 = "ZYX", i.e. Z \otimes Y \otimes X

    That is, if one runs `for i, state in enumerate(some_str)` and apply `some_str[0] \otimes some_str[1] \otimes ...`,
    this is big endian to big endian, and little endian to little endian.
    Note that the QuantumCircuit instance adopts the list of qubit indices (q_0, q_1, ...) in a big endian style.
    """
    if (endian_dm[:3] == "little" and endian_pauli[:6] == "big") or \
       (endian_dm[:6] == "big" and endian_pauli[:3] == "little"):
        str_pauli = str_pauli[::-1]

    ret = 1
    for char_pauli in str_pauli:
        if char_pauli == "I":
            ret = np.kron(ret, matrix_I)
        elif char_pauli == "X":
            ret = np.kron(ret, matrix_X)
        elif char_pauli == "Y":
            ret = np.kron(ret, matrix_Y)
        elif char_pauli == "Z":
            ret = np.kron(ret, matrix_Z)
        elif char_pauli == "0":
            ret = np.kron(ret, matrix_Zero)
        elif char_pauli == "1":
            ret = np.kron(ret, matrix_One)
        else:
            Exception
    return DMExtended(ret)


def hamiltonian_to_dm(hamiltonian: dict,
                      endian_hamiltonian: str = "big",
                      endian_dm: str = "little",
                     ) -> DMExtended:
    """
    endian depends on str_pauli_to_dm

    big endian:    q_0 = X, q_1 = Y, q_2 = Z -> q_0 q_1 q_2 = "XYZ", i.e. X \otimes Y \otimes Z
    little endian: q_0 = X, q_1 = Y, q_2 = Z -> q_2 q_1 q_0 = "ZYX", i.e. Z \otimes Y \otimes X

    That is, if one runs `for i, state in enumerate(some_str)` and apply `some_str[0] \otimes some_str[1] \otimes ...`,
    this is big endian to big endian, and little endian to little endian.
    Note that the QuantumCircuit instance adopts the list of qubit indices (q_0, q_1, ...) in a big endian style.
    """
    ret = None
    for str_pauli, coeff in hamiltonian.items():
        if ret is None:
            ret = str_pauli_to_dm(str_pauli=str_pauli,
                                  endian_dm=endian_dm,
                                  endian_pauli=endian_hamiltonian) * coeff
        else:
            ret += str_pauli_to_dm(str_pauli,
                                   endian_dm=endian_dm,
                                   endian_pauli=endian_hamiltonian) * coeff
    return ret


def dm_to_expval(density_matrix: qi.DensityMatrix, 
                 str_observable: str,
                 endian_dm: str = "little",
                 endian_observable: str = "big",
                ) -> Union[float, complex]:
    """
    big endian by default for both dm and observable
    Args:
        density_matrix: supposed to be in little endian
        str_pauli: supposed to be in big endian

    big endian:    q_0 = X, q_1 = Y, q_2 = Z -> q_0 q_1 q_2 = "XYZ", i.e. X \otimes Y \otimes Z
    little endian: q_0 = X, q_1 = Y, q_2 = Z -> q_2 q_1 q_0 = "ZYX", i.e. Z \otimes Y \otimes X

    That is, if one runs `for i, state in enumerate(some_str)` and apply `some_str[0] \otimes some_str[1] \otimes ...`,
    this is big endian to big endian, and little endian to little endian.
    Note that the QuantumCircuit instance adopts the list of qubit indices (q_0, q_1, ...) in a big endian style.
    """
    if (endian_dm[:3] == "little" and endian_observable[:6] == "big") or \
       (endian_dm[:6] == "big" and endian_observable[:3] == "little"):
        str_observable = str_observable[::-1]

    dm_observable = 1
    for char_pauli in str_observable:
        if char_pauli == "I":
            dm_observable = np.kron(dm_observable, matrix_I)
        elif char_pauli == "X":
            dm_observable = np.kron(dm_observable, matrix_X)
        elif char_pauli == "Y":
            dm_observable = np.kron(dm_observable, matrix_Y)
        elif char_pauli == "Z":
            dm_observable = np.kron(dm_observable, matrix_Z)
        elif char_pauli == "0":
            dm_observable = np.kron(dm_observable, matrix_Zero)
        elif char_pauli == "1":
            dm_observable = np.kron(dm_observable, matrix_One)
        else:
            raise Exception(char_pauli)
    density_matrix = density_matrix @ dm_observable
    return density_matrix.trace()


def dm_to_hist(density_matrix: qi.DensityMatrix, 
               endian_dm: str = "little", 
               endian_hist: str = "little",
              ) -> Counts:
    """
    density_matrix: little endian (given default)
    hist: little endian (default)

    big endian:    q_0 = X, q_1 = Y, q_2 = Z -> q_0 q_1 q_2 = "XYZ", i.e. X \otimes Y \otimes Z
    little endian: q_0 = X, q_1 = Y, q_2 = Z -> q_2 q_1 q_0 = "ZYX", i.e. Z \otimes Y \otimes X

    That is, if one runs `for i, state in enumerate(some_str)` and apply `some_str[0] \otimes some_str[1] \otimes ...`,
    this is big endian to big endian, and little endian to little endian.
    Note that the QuantumCircuit instance adopts the list of qubit indices (q_0, q_1, ...) in a big endian style.
    """
    num_qubits = int(np.log2(density_matrix.dim))
    hist = dict()
    flag_reverse_endian = 1 if endian_hist == endian_dm else -1
    # iteration start from 00...0 to 11...1
    for dm_index in range(0, density_matrix.dim):
        if np.abs(density_matrix.data[dm_index, dm_index]) > 0:
            hist[format(dm_index, "0"+str(num_qubits)+"b")[::flag_reverse_endian]] = np.abs(density_matrix.data[dm_index, dm_index])
    return Counts(hist)


def dms_to_hists(dm_list: List[qi.DensityMatrix], 
                 endian_dm: str = "little", 
                 endian_hist: str = "little",
                ) -> List[Counts]:
    """
    big endian:    q_0 = X, q_1 = Y, q_2 = Z -> q_0 q_1 q_2 = "XYZ", i.e. X \otimes Y \otimes Z
    little endian: q_0 = X, q_1 = Y, q_2 = Z -> q_2 q_1 q_0 = "ZYX", i.e. Z \otimes Y \otimes X

    That is, if one runs `for i, state in enumerate(some_str)` and apply `some_str[0] \otimes some_str[1] \otimes ...`,
    this is big endian to big endian, and little endian to little endian.
    Note that the QuantumCircuit instance adopts the list of qubit indices (q_0, q_1, ...) in a big endian style.
    """
    return [dm_to_hist(dm, endian_dm, endian_hist) for dm in dm_list]


def get_dms(result: Result,
            endian_dm: str = "little",
           ) -> List[qi.DensityMatrix]:
    """
    Note that you have to add `qc.save_density_matrix()`.
    `result.results[i].data.density_matrix` is equivalent to
    `result.data()["density_matrix"]`

    big endian:    q_0 = X, q_1 = Y, q_2 = Z -> q_0 q_1 q_2 = "XYZ", i.e. X \otimes Y \otimes Z
    little endian: q_0 = X, q_1 = Y, q_2 = Z -> q_2 q_1 q_0 = "ZYX", i.e. Z \otimes Y \otimes X

    That is, if one runs `for i, state in enumerate(some_str)` and apply `some_str[0] \otimes some_str[1] \otimes ...`,
    this is big endian to big endian, and little endian to little endian.
    Note that the QuantumCircuit instance adopts the list of qubit indices (q_0, q_1, ...) in a big endian style.
    """
    dms = []
    for experimental_result in result.results:
        dm = experimental_result.data.density_matrix
        if endian_dm == "big":
            dms.append(dm.reverse_qargs())
        else:
            dms.append(dm)
    return dms


def make_dm_binary(str_binary: str,
                   endian_binary: str = "big",
                   endian_dm: str = "little",
                  ) -> np.ndarray:
    """
    str_binary: big endian
    output: little endian

    big endian:    q_0 = X, q_1 = Y, q_2 = Z -> q_0 q_1 q_2 = "XYZ", i.e. X \otimes Y \otimes Z
    little endian: q_0 = X, q_1 = Y, q_2 = Z -> q_2 q_1 q_0 = "ZYX", i.e. Z \otimes Y \otimes X

    That is, if one runs `for i, state in enumerate(some_str)` and apply `some_str[0] \otimes some_str[1] \otimes ...`,
    this is big endian to big endian, and little endian to little endian.
    Note that the QuantumCircuit instance adopts the list of qubit indices (q_0, q_1, ...) in a big endian style.
    """
    flag_reverse_endian = 1 if endian_binary == endian_dm else -1
    str_binary = str_binary[::flag_reverse_endian]
    dm = 1 # matrix_Zero if str_binary[0] == "0" else matrix_One
    for char_binary in str_binary:
        if char_binary == "0":
            dm = np.kron(dm, matrix_Zero)
        elif char_binary == "1":
            dm = np.kron(dm, matrix_One)
        else:
            raise Exception
    return dm


def compute_distance_trace_unitary_mod_phase(U, V):
    # remove global phase
    c = np.trace(U.conj().T @ V)
    theta = np.angle(c)
    V_phase = np.exp(-1j * theta) * V

    # trace norm ||U - V_phase||_1 = sum(singular values)
    diff = U - V_phase
    s = np.linalg.svd(diff, compute_uv=False)
    return 0.5 * np.sum(s)


### ========================================================== ###


# --- 1-qubit depolarizing channel (maximally mixed at p=1) ---
def kraus_depolarizing_1q(p: float) -> qi.Kraus:
    r"""
    1-qubit depolarizing (isotropic) channel:
        E_p(ρ) = (1 - p) ρ + p * I/2
    p=1 → E_1(ρ) = I/2 (maximally mixed)
    Kraus operators:
        K0 = sqrt(a) I,  Ki = sqrt(b) P_i (P ∈ {X,Y,Z})
        a = 1 - 3p/4,  b = p/4
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1].")
    a = 1 - 3 * p / 4
    b = p / 4
    Ks = [
        np.sqrt(a) * matrix_I,
        np.sqrt(b) * matrix_X,
        np.sqrt(b) * matrix_Y,
        np.sqrt(b) * matrix_Z,
    ]
    return qi.Kraus(Ks)


# --- Utility ---
def kron(*ops):
    """Kronecker product of multiple matrices."""
    out = np.array([[1]], dtype=complex)
    for op in ops:
        out = np.kron(out, op)
    return out


def two_qubit_paulis():
    """List of 16 two-qubit Pauli operators [II, IX, IY, ..., ZZ]."""
    singles = [matrix_I, matrix_X, matrix_Y, matrix_Z]
    return [kron(a, b) for a in singles for b in singles]


# --- 2-qubit depolarizing channel (maximally mixed at p=1) ---
def kraus_depolarizing_2q(p: float) -> qi.Kraus:
    r"""
    2-qubit depolarizing (isotropic) channel:
        E_p(p) = (1 - p) p + p * I/4
    p=1 → E_1(p) = I/4 (maximally mixed)
    Kraus operators:
        K0 = sqrt(a) II,  Ki = sqrt(b) P_i (P ∈ 15 nontrivial two-qubit Paulis)
        a = 1 - 15p/16,  b = p/16
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1].")
    dim = 4
    a = 1 - (dim ** 2 - 1) * p / (dim ** 2)  # = 1 - 15p/16
    b = p / (dim ** 2)                   # = p/16
    P2 = two_qubit_paulis()
    Ks = [np.sqrt(a) * P2[0]]  # II
    Ks += [np.sqrt(b) * P for P in P2[1:]]  # 15 nontrivial Paulis
    return qi.Kraus(Ks)