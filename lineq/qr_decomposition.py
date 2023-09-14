import numpy as np
from lineq.gauss import backward_substitution

# Create the ith standard basis vector of size n
def e_n(i, n):
    e = np.zeros(n)
    e[i] = 1
    return e

# Takes as input a matrix and returns an orthogonal matrix Q
# and an upper triangular matrix R such that A = QR
def qr_decomposition(A):
    R = A.copy()
    Q = np.identity(len(A))
    for i in range(len(A[0])):
        x = R[i:, i].copy()
        u = x - np.linalg.norm(x) * e_n(0, len(x))
        if np.linalg.norm(u) == 0:
            continue
        u = u / np.linalg.norm(u)
        H = np.identity(len(x)) - 2 * (u[:, None] @ u[:, None].T)
        # Mirrors the submatrices of R until R is upper triangular
        R[i:, i:] = H @ R[i:, i:]
        # Calculates the Q matrix corresponding to R
        H_outer = np.identity(len(A))
        H_outer[i:, i:] = H
        Q = Q @ H_outer
    return Q, R

# Using the QR Decomposition algorithm
# this function solves the system Ax = b
# and returns the solution x
def qr_solve(A, b):
    Q, R = qr_decomposition(A)
    b_prim = Q.T @ b
    return backward_substitution(R, b_prim)

# Using the QR Decomposition algorithm
# this function solves the linear least squares problem Ax = b
# and returns the optimal solution x
def qr_linear_least_squares(A, b):
    Q, R = qr_decomposition(A)
    b_prim = Q.T @ b
    # Remove the extra rows from R and b_prim
    R = R[: len(R[0])]
    b_prim = b_prim[: len(R[0])]
    # Solve the system
    return backward_substitution(R, b_prim)
