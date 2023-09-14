import numpy as np
from lineq.gauss import backward_substitution


def e_n(i, n):
    # Create a standard basis vector of size n
    e = np.zeros(n)
    e[i] = 1
    return e


def qr_decomposition(A):
    R = A.copy()
    Q = np.identity(len(A))
    for i in range(len(A)):
        x = R[i:, i].copy()
        u = x - np.linalg.norm(x) * e_n(0, len(x))
        if np.linalg.norm(u) == 0:
            continue
        u = u / np.linalg.norm(u)
        H = np.identity(len(x)) - 2 * (u[:, None] @ u[:, None].T)
        # Mirrors the submatrices of R until R is upper triangular
        R[i:, i:] = (
            H
            @ R[
                i:,
                i:,
            ]
        )
        # Calculates the Q matrix corresponding to R
        H_outer = np.identity(len(A))
        H_outer[i:, i:] = H
        Q = Q @ H_outer
    return Q, R


def qr_solve(A, b):
    Q, R = qr_decomposition(A)
    b_prim = Q.T @ b
    return backward_substitution(R, b_prim)
