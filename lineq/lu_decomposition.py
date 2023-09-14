import numpy as np
from lineq.gauss import backward_substitution

# TODO: Add permutation matrix when the matrix has zeros in the diagonal

# Implementation of the LU Decomposition algorithm
# Takes as input a matrix and returns a lower and upper triangular matrices
# such that A = LU
def lu_decomposition(A):
    n = len(A)
    L = np.identity(n)
    U = np.zeros((n, n))
    for i in range(n):
        # Fill a row in the upper triangular matrix (U)
        U[i, i:] = A[i, i:]
        U[i, i:] -= L[i, :i] @ U[:i, i:]

        # Fill a column in the lower triangular matrix (L)
        L[i + 1 :, i] = A[i + 1 :, i]
        L[i + 1 :, i] -= L[i + 1 :, :i] @ U[:i, i]
        L[i + 1 :, i] = L[i + 1 :, i] / U[i][i]
    return L, U

# Takes as input a lower triangular matrix and a vector
# and returns the solution of the system
def l_backward_substitution(L_input, b_input):
    # Does normal backward substitution but for the L matrix of the LU decomposition
    singular = np.any(np.all(L_input == 0, axis=1))
    if singular:
        raise Exception("Matrix is singular")
    L = L_input.copy()
    b = b_input
    n = L.shape[0]
    x = np.zeros(n)
    for var in range(n):
        x[var] = b[var] / L[var, var]
        b = (b.T - (L[:, var] * x[var]).T).T
        L[:, var] = np.zeros(n)
    return x

# Using the LU Decomposition algorithm
# this function solves the system Ax = b
# in two steps:
# 1. LUx = b
# 2. Ux = y
# 3. Ly = b
# and returns the solution x
def lu_solve(A, b):
    L, R = lu_decomposition(A)
    y = l_backward_substitution(L, b)
    x = backward_substitution(R, y)
    return x
