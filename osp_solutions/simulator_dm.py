from typing import *
import numpy as np
import copy
from qiskit.circuit import QuantumCircuit
from qiskit_aer.primitives import SamplerV2 as AerSampler
import qiskit.quantum_info as qi
from qiskit.result import Counts, Result #! deprecated 


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
             flip_endian: bool = False,
            ) -> DMExtended:
    """
    Note that you have to add `qc.save_density_matrix()`.
    `result.data()["density_matrix"]` is equivalent to
    `result.results[i].data.density_matrix`
    #! deprecated library: qiskit.aer
    big endian by default
    """
    sampler = AerSampler() ### here, backend is not used ###
    job = sampler.run(qc, shots=0)
    result = job.result()
    dm = result.data()["density_matrix"]
    if flip_endian:
        return DMExtended(dm.reverse_qargs())
    else:
        return DMExtended(dm)


def str_pauli_to_dm(str_pauli: Union[str, qi.Pauli],
                    endian_pauli: str = "little",
                    endian_dm: str = "little",
                   ) -> DMExtended:
    """
    little endian by default
    """
    if (endian_dm[:3] == "big" and endian_pauli[:6] == "little") or \
       (endian_dm[:6] == "little" and endian_pauli[:3] == "big"):
        str_pauli = str_pauli[::-1]

    ret = 1
    for char_pauli in str_pauli:
        if char_pauli == "I":
            dm = np.array([[1, 0], [0, 1]], dtype="complex")
        elif char_pauli == "X":
            dm = np.array([[0, 1], [1, 0]], dtype="complex")
        elif char_pauli == "Y":
            dm = np.array([[0, -1j], [1j, 0]], dtype="complex")
        elif char_pauli == "Z":
            dm = np.array([[1, 0], [0, -1]], dtype="complex")
        elif char_pauli == "0":
            dm = np.array([[1, 0], [0, 0]], dtype="complex")
        elif char_pauli == "1":
            dm = np.array([[0, 0], [0, 1]], dtype="complex")
        else:
            Exception
        ret = np.kron(ret, dm)
    return DMExtended(ret)


def hamiltonian_to_dm(hamiltonian: dict,
                      endian_hamiltonian: str = "little",
                      endian_dm: str = "little",
                     ) -> DMExtended:
    """
    endian depends on str_pauli_to_dm
    little endian by default
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
                 endian_dm: str = "little_endian",
                 endian_observable: str = "little_endian",
                ) -> Union[float, complex]:
    """
    little endian by default for both dm and observable
    Args:
        density_matrix: supposed to be in little endian
        str_pauli: supposed to be in little endian
    """
    if (endian_dm[:3] == "big" and endian_observable[:6] == "little") or \
       (endian_dm[:6] == "little" and endian_observable[:3] == "big"):
        str_observable = str_observable[::-1]

    dm_observable = 1
    for char_pauli in str_observable:
        if char_pauli == "I":
            dm_observable = np.kron(dm_observable, 
                                    np.array([[1, 0],
                                              [0, 1]],
                                    dtype="complex"))
        elif char_pauli == "X":
            dm_observable = np.kron(dm_observable, 
                                    np.array([[0, 1],
                                              [1, 0]],      
                                    dtype="complex"))
        elif char_pauli == "Y":
            dm_observable = np.kron(dm_observable, 
                                    np.array([[0, -1j],
                                              [1j, 0]],
                                    dtype="complex"))
        elif char_pauli == "Z":
            dm_observable = np.kron(dm_observable, 
                                    np.array([[1, 0],
                                              [0, -1]],
                                    dtype="complex"))
        elif char_pauli == "0":
            dm_observable = np.kron(dm_observable, 
                                    np.array([[1, 0],
                                              [0, 0]],
                                    dtype="complex"))
        elif char_pauli == "1":
            dm_observable = np.kron(dm_observable, 
                                    np.array([[0, 0],
                                              [0, 1]],
                                    dtype="complex"))
        else:
            raise Exception(char_pauli)
    density_matrix = density_matrix @ dm_observable
    return density_matrix.trace()


def dm_to_hist(density_matrix: qi.DensityMatrix, 
               endian_dm: str = "big_endian", 
               endian_hist: str = "big_endian",
              ) -> Counts:
    """
    density_matrix: big_endian (given default)
    hist: big_endian (default)
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
                 endian_dm: str = "big_endian", 
                 endian_hist: str = "big_endian",
                ) -> List[Counts]:
    return [dm_to_hist(dm, endian_dm, endian_hist) for dm in dm_list]


def get_dms(result: Result,
            endian_dm: str = "big_endian",
           ) -> List[qi.DensityMatrix]:
    """
    Note that you have to add `qc.save_density_matrix()`.
    `result.results[i].data.density_matrix` is equivalent to
    `result.data()["density_matrix"]`
    """
    dms = []
    for experimental_result in result.results:
        dm = experimental_result.data.density_matrix
        if endian_dm == "little" or endian_dm == "little_endian":
            dms.append(dm.reverse_qargs())
        else:
            dms.append(dm)
    return dms


def make_dm_binary(str_binary: str,
                   endian_binary: str = "little_endian", 
                   endian_dm: str = "big_endian",
                  ) -> np.ndarray:
    """
    str_binary: little endian
    output: big endian
    """
    Zero = np.array([[1,0],
                     [0,0]])
    One = np.array([[0,0],
                    [0,1]])
    flag_reverse_endian = 1 if endian_binary == endian_dm else -1
    str_binary = str_binary[::flag_reverse_endian]
    dm = Zero if str_binary[0] == "0" else One
    for char_binary in str_binary[1:]:
        if char_binary == "0":
            dm = np.kron(dm, Zero)
        elif char_binary == "1":
            dm = np.kron(dm, One)
        else:
            raise Exception
    return dm