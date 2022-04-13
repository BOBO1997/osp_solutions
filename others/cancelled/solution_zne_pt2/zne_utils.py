import re
import itertools
import numpy as np
import random
random.seed(42)
import mitiq
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.ignis.mitigation import expectation_value


# Pauli Twirling
def pauli_twirling(circ) -> QuantumCircuit:
    #! qasm ベタ書き
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
    ops = circ.qasm().splitlines(True) #! 生のqasmコードを持ってきてる: オペレータに分解
    for op in ops:
        if (op[:2] == 'cx'): # can add for cz, etc.
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


def zne_wrapper(qcs, scale_factors = [1.0, 2.0, 3.0]):
    """
    outputs the circuit list for zero noise extrapolation
    WITHOUT Pauli twirling
    """
    folded_qcs = [] #! ZNE用の回路
    for qc in qcs:
        folded_qcs.append([mitiq.zne.scaling.fold_gates_at_random(qc, scale) for scale in scale_factors]) #! ここでmitiqを使用
    folded_qcs = list(itertools.chain(*folded_qcs)) #! folded_qcsを平坦化
    folded_qcs = [pauli_twirling(circ) for circ in folded_qcs]
    return folded_qcs


def make_stf_basis(n, basis_elements = ["X","Y","Z"]):
    if n == 1:
        return basis_elements
    basis = []
    for i in basis_elements:
        sub_basis = make_stf_basis(n - 1, basis_elements)
        basis += [i + j for j in sub_basis]
    return basis


def reduce_hist(hist, poses):
    n = len(poses)
    ret_hist = {format(i, "0" + str(n) + "b"): 0 for i in range(1 << n)}
    for k, v in hist.items():
        pos = ""
        for i in range(n):
            pos += k[poses[i]]
        ret_hist[pos] += v
    return ret_hist


def make_stf_expvals(n, stf_hists):
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


def zne_decoder(n, result, scale_factors=[1.0, 2.0, 3.0]):
    
    hists = result.get_counts()
    num_scale_factors = len(scale_factors)
    assert len(hists) % num_scale_factors == 0
    scale_wise_expvals = [] # num_scale_factors * 64
    for i in range(num_scale_factors):
        scale_wise_hists = [hists[3 * j + i] for j in range(len(hists) // num_scale_factors)]
        st_expvals = make_stf_expvals(n, scale_wise_hists)
        scale_wise_expvals.append( list(st_expvals.values()) )
    
    scale_wise_expvals = np.array(scale_wise_expvals)
    
    fac = mitiq.zne.inference.LinearFactory(scale_factors)
    zne_expvals = []
    for i in range(4 ** n):
        zne_expvals.append( fac.extrapolate(scale_factors, scale_wise_expvals[:, i]) )
        # or we can use the following exp factory
        # zne.ExpFactory.extrapolate(scale_factors, expectation_values)
    
    return zne_expvals