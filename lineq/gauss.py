import numpy as np


# TODO: Check if having zeros in the diagonal leads to undefined behavior
def backward_substitution(A, b):
    # Copy the matrix so that the original is not modified
    singular = np.any(np.all(A == 0, axis=1))
    if singular:
        raise Exception("Matrix is singular")
    A_copy = A.copy()
    b_copy = b.copy()
    n = A.shape[0]
    x = np.zeros(n)
    for var in range(n - 1, -1, -1):
        x[var] = b_copy[var] / A_copy[var, var]
        b_copy = (b_copy.T - (A_copy[:, var] * x[var]).T).T
        A_copy[:, var] = np.zeros(n)
    return x


def forward_substitution(A, b):
    n = A.shape[0]
    A_copy = A.copy()
    b_copy = b.copy()
    for var in range(n - 1):
        # Find the pivot for the needed column and swap it
        pivot_row = np.argmax(abs(A[var:, var])) + var
        A_copy[var], A_copy[pivot_row] = A_copy[pivot_row].copy(), A_copy[var].copy()
        b_copy[var], b_copy[pivot_row] = b_copy[pivot_row].copy(), b_copy[var].copy()
        # Subtract the first row from a multiple of the rest of the rows
        non_zero_indices = np.where(A_copy[var + 1 :, var] != 0)
        coeff = (A_copy[var, var] / A_copy[var + 1 :, var][non_zero_indices]).copy()
        A_copy[var + 1 :,][non_zero_indices] = (
            A_copy[var + 1 :][non_zero_indices].T * coeff
        ).T - A_copy[var]
        b_copy[var + 1 :,][non_zero_indices] = (
            b_copy[var + 1 :][non_zero_indices].T * coeff
        ).T - b_copy[var]
    return A_copy, b_copy


def gauss_elimination(A, b):
    assert A.shape[0] == A.shape[1], "Matrix A must be square"
    assert A.shape[0] == b.shape[0], "Matrix A and b must have the same number of rows"
    A, b = forward_substitution(A, b)
    singular = np.any(np.all(A == 0, axis=1))
    if singular:
        raise Exception("Matrix is singular")
    x = backward_substitution(A, b)
    return x
