from typing import *
import numpy as np
import copy
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator
# from qiskit_aer.primitives import SamplerV2 as AerSampler
import qiskit.quantum_info as qi
from qiskit.result import Counts, Result #! deprecated 


class DMExtended(qi.DensityMatrix):
    def __init__(self, 
                 matrix, 
                 dims=None,
                ) -> None:
        super().__init__(matrix, dims=dims)

    def __matmul__(self, 
                   other,
                  ) -> Any:
        if isinstance(other, qi.DensityMatrix):
            return DMExtended(self._data @ other._data, dims=self.dims())
        else:
            return DMExtended(self._data @ other, dims=self.dims())
        
    def __pow__(self, 
                num_power: int,
               ) -> Any:
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
    
    def dagger(self) -> Any:
        return DMExtended(self._data.T.conjugate(), dims=self.dims())
    
    def tensor_pow(self, 
                   k: int,
                  ) -> Any:
        if k <= 0:
            return 0
        elif k == 1:
            return DMExtended(self._data, dims=self.dims())
        else:
            return DMExtended(self._data, dims=self.dims()) ^ DMExtended(self._data, dims=self.dims()).tensor_pow(k - 1)

    def normalize(self):
        return DMExtended(self._data / self.trace(), dims=self.dims())
    
    def partial_trace(self, 
                      qargs: List[int],
                      endian_dm: int = "little", ### for my own use, it's usually little endian ###
                      endian_qargs: int = "little",
                      normalize: bool = True,
                     ) -> Any:
        """
        qubtis: the subsystem to be traced over <- indices denoted in little endian but the denisty matrix is big endian by default in qi.DensityMatrix
        with normalization by default
        """
        if endian_dm == endian_qargs: ### flip endian of qargs since qi.partial trace takes state in big endiang and qargs in little endian ###
            qargs = [self.num_qubits - 1 - qarg for qarg in qargs]
        if normalize:
            return DMExtended(qi.partial_trace(state=self._data, qargs=qargs)).normalize()
        else:
            return DMExtended(qi.partial_trace(state=self._data, qargs=qargs))
    
    def apply_unitary(self, 
                      U: qi.DensityMatrix, 
                      normalize: bool = True,
                     ) -> Any:
        """
        [with normalization]
        """
        ret_raw = U @ self @ U.dagger()
        if normalize:
            return DMExtended(U @ self @ U.dagger()).normalize()
        else:
            return DMExtended(U @ self @ U.dagger())


### functions for converting quantum info to density matrice ###

def qc_to_dm(qc: QuantumCircuit,
             noise_model: Any = None,
             flip_endian: bool = False,
            ) -> DMExtended:
    """
    Note that you have to add `qc.save_density_matrix()`.
    `result.data()["density_matrix"]` is equivalent to
    `result.results[i].data.density_matrix`
    #! deprecated library: qiskit.aer
    - big endian by default
    - initial state is |0>
    """
    simulator = AerSimulator(method="density_matrix",
                             noise_model=noise_model) ### here, backend is not used ###
    job = simulator.run(circuits=qc,
                        shots=0)
    result = job.result()
    dm = result.data()["density_matrix"]
    if flip_endian:
        return DMExtended(dm.reverse_qargs())
    else:
        return DMExtended(dm)


def str_pauli_to_dm(str_pauli: Union[str, qi.Pauli],
                    endian_dm: str = "little",
                    endian_pauli: str = "little",
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


def str_pauli_to_dm_opr_meas(str_pauli: Union[str, qi.Pauli],
                             endian_dm: str = "little",
                             endian_pauli: str = "little",
                            ) -> DMExtended:
    """
    little endian by default
    """
    if (endian_dm[:3] == "big" and endian_pauli[:6] == "little") or \
       (endian_dm[:6] == "little" and endian_pauli[:3] == "big"):
        str_pauli = str_pauli[::-1]

    dm_I = np.array([[1, 0], [0, 1]], dtype="complex")
    dm_H = np.array([[1, 1], [1, -1]], dtype="complex") / np.sqrt(2)
    dm_S = np.array([[1, 0], [0, 1j]], dtype="complex")
    dm_S_dagger = np.array([[1, 0], [0, -1j]], dtype="complex")

    ret = 1
    for char_pauli in str_pauli:
        if char_pauli == "I":
            dm = dm_I
        elif char_pauli == "X":
            dm = dm_H
        elif char_pauli == "Y":
            dm = dm_H @ dm_S_dagger
        elif char_pauli == "Z":
            dm = dm_I
        else:
            Exception
        ret = np.kron(ret, dm)
    return DMExtended(ret)


def hamiltonian_to_dm(hamiltonian: dict,
                      endian_dm: str = "little",
                      endian_hamiltonian: str = "little",
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


### functions for density matrices to measurement results ###

def dm_to_expval(dm: qi.DensityMatrix, 
                 str_observable: str,
                 endian_dm: str = "little",
                 endian_observable: str = "little",
                ) -> Union[float, complex]:
    """
    little endian by default for both dm and observable
    Args:
        dm: supposed to be in little endian
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
    dm = dm @ dm_observable
    return dm.trace()


def dm_to_hist(dm: qi.DensityMatrix,
               observable: str,
               endian_dm: str = "big",
               endian_observable: str = "little",
               endian_hist: str = "big",
              ) -> Counts:
    """
    ###! using dm_with_observable_to_hist is recommended.
    Here observables just indicate which qubits to measure.
    It takes care of only whether the char_pauli is I or not.
    Args:
        dm: big endian (given default)
        hist: big endian (default)
    """
    flag_reverse_endian_hist = 1 if endian_hist == endian_dm else -1 ### 1 is to do nothing, -1 is to flip the list
    flag_reverse_endian_observable = 1 if endian_observable == endian_dm else -1 ### not in use, since this effect can be absorbed in dm.partial_trace(endian_qargs=...)
    num_qubits = dm.num_qubits
    assert num_qubits == len(observable)
    indices_reduced = [i for i in range(num_qubits) if observable[i] == "I"] ### same endian as endian_observable ###
    dm_reduced = dm.partial_trace(qargs=indices_reduced, 
                                  endian_dm=endian_dm,
                                  endian_qargs=endian_observable)
    num_clbits = dm_reduced.num_qubits

    hist = dict()
    # iteration start from 00...0 to 11...1
    for index_dm in range(0, dm_reduced.dim):
        prob = np.abs(dm_reduced.data[index_dm, index_dm])
        if not np.allclose(a=prob, b=0, atol=1e-9): ### to exclude elements with prob 0 ###
            str_state = format(index_dm, "0"+str(num_clbits)+"b")[::flag_reverse_endian_hist]
            hist[str_state] = prob
    return Counts(hist)


def dms_to_hists(dms: List[qi.DensityMatrix],
                 observables: str,
                 endian_dm: str = "big",
                 endian_observable: str = "little",
                 endian_hist: str = "big",
                ) -> List[Counts]:
    """
    Here observables just indicate which qubits to measure.
    It takes care of only whether the char_pauli is I or not.
    """
    assert len(observables) == len(dms)
    hists = []
    for dm, observable in zip(dms, observables):
        hist = dm_to_hist(dm=dm, 
                          observable=observable,
                          endian_dm=endian_dm,
                          endian_observable=endian_observable,
                          endian_hist=endian_hist)
        hists.append(hist)
    return hists


def dm_with_observable_to_hist(dm: qi.DensityMatrix,
                               observable: str,
                               endian_dm: str = "big",
                               endian_observable: str = "little",
                               endian_hist: str = "big",
                              ) -> Counts:
    flag_reverse_endian_hist = 1 if endian_hist == endian_dm else -1
    flag_reverse_endian_observable = 1 if endian_observable == endian_dm else -1
    num_qubits = dm.num_qubits
    assert num_qubits == len(observable)
    dm_observable = str_pauli_to_dm_opr_meas(str_pauli=observable[::flag_reverse_endian_observable],
                                             endian_dm=endian_dm,
                                             endian_pauli=endian_dm) ### endian should be the same here ###
    dm_with_observable = dm_observable @ dm @ dm_observable.dagger()
    indices_reduced = [i for i in range(num_qubits) if observable[i] == "I"] ### same endian as endian_observable ###
    dm_reduced = dm_with_observable.partial_trace(qargs=indices_reduced, 
                                                  endian_dm=endian_dm,
                                                  endian_qargs=endian_observable)
    num_clbits = dm_reduced.num_qubits

    hist = dict()
    # iteration start from 00...0 to 11...1
    for index_dm in range(0, dm_reduced.dim):
        prob = np.abs(dm_reduced.data[index_dm, index_dm])
        if not np.allclose(a=prob, b=0, atol=1e-9): ### to exclude elements with prob 0 ###
            str_state = format(index_dm, "0"+str(num_clbits)+"b")[::flag_reverse_endian_hist]
            hist[str_state] = prob
    return Counts(hist)


def dms_with_obserbables_to_hists(dms: List[qi.DensityMatrix],
                                  observables: List[str],
                                  endian_dm: str = "big",
                                  endian_observable: str = "little",
                                  endian_hist: str = "big",
                                 ) -> List[Counts]:
    assert len(observable) == len(dms)
    hists = []
    for dm, observable in zip(dms, observables):
        hist = dm_with_observable_to_hist(dm=dm, 
                                          observable=observable,
                                          endian_dm=endian_dm,
                                          endian_observable=endian_observable,
                                          endian_hist=endian_hist)
        hists.append(hist)
    return hists


def result_to_dms(result: Result,
                  endian_dm: str = "big",
                 ) -> List[DMExtended]:
    """
    Note that you have to add `qc.save_density_matrix()`.
    `result.results[i].data.density_matrix` is equivalent to
    `result.data()["density_matrix"]`
    """
    dms = []
    for experimental_result in result.results:
        dm = DMExtended(experimental_result.data.density_matrix)
        if endian_dm == "little" or endian_dm == "little_endian":
            dms.append(dm.reverse_qargs())
        else:
            dms.append(dm)
    return dms