import numpy as np
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import is_positive_semidefinite_matrix
from qiskit.quantum_info import state_fidelity
from zne_utils import make_stf_basis
from sgs_algorithm import sgs_algorithm


def pauli_matrix(basis):
    """
    リトルエンディアン
    """
    matrix = np.ones(1, dtype="complex")
    for b in basis:
        if b == "I":
            matrix = np.kron(matrix, np.array([[1,0],
                                               [0,1]], dtype="complex"))
        elif b == "X":
            matrix = np.kron(matrix, np.array([[0,1],
                                               [1,0]], dtype="complex"))
        elif b == "Y":
            matrix = np.kron(matrix, np.array([[0,-1j],
                                               [1j,0]], dtype="complex"))
        elif b == "Z":
            matrix = np.kron(matrix, np.array([[1,0],
                                               [0,-1]], dtype="complex"))
        else:
            raise Exception
    return matrix

def reconstruct_density_matrix(n, expvals):
    """
    エンディアンに注意
    """
    st_basis = make_stf_basis(n, basis_elements=["I","X","Y","Z"])
    rho = np.zeros((1 << n, 1 << n), dtype="complex")
    for expval, basis in zip(expvals, st_basis):
        rho += expval * pauli_matrix(basis[::-1]) # [::-1]でbig endianとlittle endianを交換する # 今の所、[::-1]がbig endianで、計算結果が正しそう
    rho /= (1 << n)
    return rho

def fit_valid_rho(rho):
    eigs, P = np.linalg.eig(rho) # あやしい <================================
    eigs_dict = {i: eig for i, eig in enumerate(np.real(eigs))} # あやしい <================================
    eigs_fit_dict = sgs_algorithm(eigs_dict) # あやしい <================================
    eigs_fit = np.array(list(eigs_fit_dict.values()), dtype="complex")[np.argsort(list(eigs_fit_dict.keys()))]
    rho_fit = P @ np.diag(eigs_fit) @ P.T.conjugate()
    if not np.allclose(np.trace(rho_fit), 1.0):
        print(np.trace(rho_fit))
        rho_fit /= np.trace(rho_fit) # 無理矢理!!!!!!! <=================================
    return rho_fit

def expvals_to_valid_rho(n, zne_expvals, assertion=True):
    rho = reconstruct_density_matrix(n, zne_expvals)
    rho_fit = fit_valid_rho(rho)
    if assertion:
        assert np.allclose(np.trace(rho_fit), 1.0)
        assert is_hermitian_matrix(rho_fit)
        assert is_positive_semidefinite_matrix(rho_fit)
    return rho_fit