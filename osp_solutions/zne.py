import re
import itertools
import numpy as np
import random
random.seed(42)
import mitiq
from qiskit import QuantumCircuit, QuantumRegister
from osp_solutions.expval_ignis import expectation_value


# Pauli Twirling
def pauli_twirling(circ: QuantumCircuit) -> QuantumCircuit:
    """
    [internal function]
    
    This function takes a quantum circuit and return a new quantum circuit with Pauli Twirling around the CNOT gates.
    Args:
        circ: QuantumCircuit
    Returns:
        QuantumCircuit
    """
    def apply_pauli(num: int, qb: int) -> str:
        if (num == 0):
            return f''
        elif (num == 1):
            return f'x q[{qb}];\n'
        elif (num == 2):
            return f'y q[{qb}];\n'
        else:
            return f'z q[{qb}];\n'

    paulis = [(i,j) for i in range(0,4) for j in range(0,4)]
    paulis.remove((0,0))
    paulis_map = [(0, 1), (3, 2), (3, 3), (1, 1), (1, 0), (2, 3), (2, 2), (2, 1), (2, 0), (1, 3), (1, 2), (3, 0), (3, 1), (0, 2), (0, 3)]

    new_circ = ''
    ops = circ.qasm().splitlines(True) #! split the quantum circuit into qasm operators
    for op in ops:
        if (op[:2] == 'cx'): # add Pauli Twirling around the CNOT gate
            num = random.randrange(len(paulis))
            qbs = re.findall('q\[(.)\]', op)
            new_circ += apply_pauli(paulis[num][0], qbs[0])
            new_circ += apply_pauli(paulis[num][1], qbs[1])
            new_circ += op
            new_circ += apply_pauli(paulis_map[num][0], qbs[0])
            new_circ += apply_pauli(paulis_map[num][1], qbs[1])
        else:
            new_circ += op
    return QuantumCircuit.from_qasm_str(new_circ)


def zne_wrapper(qcs, scale_factors = [1.0, 2.0, 3.0], pt = False):
    """
    This function outputs the circuit list for zero-noise extrapolation.
    Args:
        qcs: List[QuantumCircuit], the input quantum circuits.
        scale_factors: List[float], to what extent the noise scales are investigated.
        pt: bool, whether add Pauli Twirling or not.
    Returns:
        folded_qcs: List[QuantumCircuit]
    """
    folded_qcs = [] #! ZNE用の回路
    for qc in qcs:
        folded_qcs.append([mitiq.zne.scaling.fold_gates_at_random(qc, scale) for scale in scale_factors]) #! ここでmitiqを使用
    folded_qcs = list(itertools.chain(*folded_qcs)) #! folded_qcsを平坦化
    if pt:
        folded_qcs = [pauli_twirling(circ) for circ in folded_qcs]
    return folded_qcs


def make_stf_basis(n, basis_elements = ["X","Y","Z"]):
    """
    [internal function]
    
    This function outputs all the combinations of length n string for given basis_elements.
    When basis_elements is X, Y, and Z (default), the output becomes the n-qubit Pauli basis.
    Args:
        n: int
        basis_elements: List[str]
    Returns:
        basis: List[str]
    """
    if n == 1:
        return basis_elements
    basis = []
    for i in basis_elements:
        sub_basis = make_stf_basis(n - 1, basis_elements)
        basis += [i + j for j in sub_basis]
    return basis


def reduce_hist(hist, poses):
    """
    [internal function]
    
    This function returns the reduced histogram to the designated positions.
    Args:
        hist: Dict[str, float]
        poses: List[int]
    Returns:
        ret_hist: Dict[str, float]
    """
    n = len(poses)
    ret_hist = {format(i, "0" + str(n) + "b"): 0 for i in range(1 << n)}
    for k, v in hist.items():
        pos = ""
        for i in range(n):
            pos += k[poses[i]]
        ret_hist[pos] += v
    return ret_hist


def make_stf_expvals(n, stf_hists):
    """
    [internal function]
    
    This function create the expectations under expanded basis, which are used to reconstruct the density matrix.
    Args:
        n: int, the size of classical register in the measurement results.
        stf_hists: List[Dict[str, float]], the input State Tomography Fitter histograms.
    Returns:
        st_expvals: List[float], the output State Tomography expectation values.
    """
    assert len(stf_hists) == 3 ** n
    stf_basis = make_stf_basis(n, basis_elements=["X","Y","Z"])
    st_basis = make_stf_basis(n, basis_elements=["I","X","Y","Z"])
    
    stf_hists_dict = {basis: hist for basis, hist in zip(stf_basis, stf_hists)}
    st_hists_dict = {basis: stf_hists_dict.get(basis, None) for basis in st_basis}
    
    # remaining
    for basis in sorted(set(st_basis) - set(stf_basis)):
        if basis == "I" * n:
            continue
        reduction_poses = []
        reduction_basis = ""
        for i, b in enumerate(basis):
            if b != "I":
                reduction_poses.append(n - 1 - i) # big endian 
                reduction_basis += b # こっちはそのまま(なぜならラベルはlittle endianだから)
            else:
                reduction_basis += "Z"
        
        st_hists_dict[basis] = reduce_hist(stf_hists_dict[reduction_basis], reduction_poses)
    
    st_expvals = dict()
    for basis, hist in st_hists_dict.items():
        if basis == "I" * n:
            st_expvals[basis] = 1.0
            continue
        st_expvals[basis], _ = expectation_value(hist)
    return st_expvals


def zne_decoder(n, result, scale_factors=[1.0, 2.0, 3.0], fac_type="lin"):
    """
    This function applies the zero-noise extrapolation to the measured results and output the mitigated zero-noise expectation values.
    Args:
        n: int, the size of classical register in the measurement results.
        result: Result, the returned results from job.
        scale_factors: List[float], this should be the same as the zne_wrapper.
        fac_type: str, "lin" or "exp", whether to use LinFactory option or ExpFactory option in mitiq, to extrapolate the expectation values.
    Returns:
        zne_expvals: List[float], the mitigated zero-noise expectation values.
    """
    hists = result.get_counts()
    num_scale_factors = len(scale_factors)
    assert len(hists) % num_scale_factors == 0
    scale_wise_expvals = [] # num_scale_factors * 64
    for i in range(num_scale_factors):
        scale_wise_hists = [hists[3 * j + i] for j in range(len(hists) // num_scale_factors)]
        st_expvals = make_stf_expvals(n, scale_wise_hists)
        scale_wise_expvals.append( list(st_expvals.values()) )
    
    scale_wise_expvals = np.array(scale_wise_expvals)
    
    linfac = mitiq.zne.inference.LinearFactory(scale_factors)
    expfac = mitiq.zne.ExpFactory(scale_factors)
    zne_expvals = []
    for i in range(4 ** n):
        if fac_type == "lin":
            zne_expvals.append( linfac.extrapolate(scale_factors, scale_wise_expvals[:, i]) )
        else:
            zne_expvals.append( expfac.extrapolate(scale_factors, scale_wise_expvals[:, i]) )
    
    return zne_expvals