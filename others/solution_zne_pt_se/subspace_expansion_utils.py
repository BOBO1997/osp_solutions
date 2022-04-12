import numpy as np
import scipy.linalg as sla


def prepare_H(A, H, sigma_list):
    H_tilde = np.zeros((len(sigma_list), len(sigma_list)), dtype="complex")
    for i in range(len(sigma_list)):
        for j in range(len(sigma_list)):
            # Tr[A * sigma_i * H * sigma_j]
            H_tilde[i, j] = (A @ sigma_list[i] @ H @
                             dag(sigma_list[j])).trace()
    return H_tilde


def prepare_S(A, sigma_list):
    S_tilde = np.zeros((len(sigma_list), len(sigma_list)), dtype="complex")
    for i in range(len(sigma_list)):
        for j in range(len(sigma_list)):
            # Tr[A * sigma_i * sigma_j]
            S_tilde[i, j] = (A @ sigma_list[i] @ dag(sigma_list[j])).trace()
    return S_tilde


def subspace_expansion(n, ):
    """
    """

def generalized_subspace_expansion(n, Hamiltonian, rho, sigma_list):
    H_tilde = prepare_H(rho, Hamiltonian, sigma_list)
    S_tilde = prepare_S(rho, sigma_list)
    # generalized eigenvalue problem
    eigvals, eigvecs = sla.eig(H_tilde, S_tilde)
    E, c_list = eigvals[0], eigvecs[0]
    scaling = (np.conjugate(c_list.reshape(
        [1, c_list.shape[0]])) @ S_tilde @ c_list.reshape([c_list.shape[0], 1]))[0, 0]
    c_list /= np.sqrt(scaling)
    P = np.zeros((1 << n, 1 << n), dtype="complex")
    for c, sigma in zip(c_list, sigma_list):
        P += c * sigma
    rho_se = P @ rho @ dag(P)
    rho_se /= rho_se.trace()
    return rho_se
