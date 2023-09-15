import numpy as np


# Implementation of some iterative methods for solving linear systems


# Jacobi iteration
def jacobi_step(A, b, x):
    M_inv = np.diag(1 / A.diagonal())
    x = x + (M_inv @ (b - (A @ x)))
    return x


def jacobi_iteration(A, b, x0=None, steps=100):
    if x0 is None:
        x0 = np.zeros((len(A), 1))
    # Diagonal of A must be non-zero
    x = x0
    for i in range(steps):
        x = jacobi_step(A, b, x)
    return x


# Gauss-Seidel iteration
def gauss_seidel_step(A, b, x):
    for i in range(len(x)):
        x[i] = x[i] + (1 / A[i, i]) * (b[i] - (A[i, :i] @ x[:i]) - (A[i, i:] @ x[i:]))
    return x


def gauss_seidel_iteration(A, b, x0=None, steps=100):
    if x0 is None:
        x0 = np.zeros((len(A), 1))
    # Diagonal of A must be non-zero
    x = x0
    for i in range(steps):
        x = gauss_seidel_step(A, b, x)
    return x


# Steepest descent method
def steepest_descent_step(A, b, x):
    residual = b - A @ x
    step_size = (residual.T @ residual) / (residual.T @ A @ residual)
    x = x + step_size * residual
    return x


def steepest_descent(A, b, x=None, steps=100):
    if x is None:
        x = np.zeros((A.shape[0], 1))
    for i in range(steps):
        x = steepest_descent_step(A, b, x)
    return x


# Power iteration method for finding the largest eigenvector of a matrix
def power_iteration_step(A, v, gamma):
    return (A - gamma * np.identity(A.shape[0])) @ v


def power_iteration(A, v=None, gamma=0.0, steps=10):
    if v is None:
        v = np.random.rand(A.shape[0])
    for i in range(steps):
        v = power_iteration_step(A, v, gamma)
    v = v / np.linalg.norm(v)
    return v


def get_eigenvalue(A, v):
    return (v @ (A @ v)) / (v @ v)
