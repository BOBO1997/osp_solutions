import numpy as np
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import is_positive_semidefinite_matrix
# from qiskit.quantum_info import state_fidelity
from osp_solutions.zne import make_stf_basis
from osp_solutions.sgs_algorithm import sgs_algorithm


def pauli_matrix(basis):
    """
    [internal function, support function of reconstruct_density_matrix]
    
    This function takes one basis element and outputs the corresponding matrix.
    Here the input basis elemnt is given in little endian.
    Args:
        basis: str
    Returns:
        matrix: np.array
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
    [internal function, support function of expvals_to_valid_rho]
    
    This function returns the valid denisity matrix constructed from the expectation values for each basis elements.
    This is based on the equatoin (8.149) in QCQI.
    Args:
        n: int, the size of classical register for the measured bitstrings.
        zne_expvals: List[float], input expectation values.
    Returns:
        rho: np.array, the density matrix reconstructed from the input expectation values.
    """
    st_basis = make_stf_basis(n, basis_elements=["I","X","Y","Z"])
    rho = np.zeros((1 << n, 1 << n), dtype="complex")
    for expval, basis in zip(expvals, st_basis):
        rho += expval * pauli_matrix(basis[::-1]) # convert the endian by [::-1] from little endian to big endian
    rho /= (1 << n)
    return rho


def fit_valid_rho(rho):
    """
    [internal function, support function of expvals_to_valid_rho]
    
    This function fit the reconstructed density matrix into the valid denisity matrix.
    We use the maximum likelihood method by Smolin, Gambetta, and Smith, 2012, PRL (SGS algorithm) to find a closet physically valid density matrix to the input density matrix.
    Args:
        rho: np.array, input density matrix which might not be physically valid.
    Returns:
        rho_fit: np.array, the density matrix reconstructed from the input expectation values.
    """
    eigs, P = np.linalg.eig(rho)
    eigs_dict = {i: eig for i, eig in enumerate(np.real(eigs))} # Here we assume the diagonal element is almost real
    eigs_fit_dict = sgs_algorithm(eigs_dict) # Apply SGS algorithm by Smolin, Gambetta, and Smith, 2012
    eigs_fit = np.array(list(eigs_fit_dict.values()), dtype="complex")[np.argsort(list(eigs_fit_dict.keys()))]
    rho_fit = P @ np.diag(eigs_fit) @ P.T.conjugate()
    if not np.allclose(np.trace(rho_fit), 1.0):
        print(np.trace(rho_fit))
        rho_fit /= np.trace(rho_fit) # rescale the trace forcibly
    return rho_fit


def expvals_to_valid_rho(n, zne_expvals, assertion=True):
    """
    This function returns the valid denisity matrix constructed from the expectation values for each basis elements.
    Args:
        n: int, the size of classical register for the measured bitstrings.
        zne_expvals: List[float], input expectation values.
        assertion: bool, whether to check the constructed density matrix is a valid one or not.
    Returns:
        rho_fit: np.array, the valid density matrix reconstructed from the input expectation values.
    """
    rho = reconstruct_density_matrix(n, zne_expvals)
    rho_fit = fit_valid_rho(rho)
    if assertion:
        assert np.allclose(np.trace(rho_fit), 1.0)
        assert is_hermitian_matrix(rho_fit)
        assert is_positive_semidefinite_matrix(rho_fit)
    return rho_fit