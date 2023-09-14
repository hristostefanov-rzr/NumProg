import numpy as np

from misc.polynom import create_base_polynom, merge_polynoms
from lineq.gauss import gauss_elimination


def create_polynom_linear_equation(xs, base_polynoms):
    return np.array([base_polynom(xs) for base_polynom in base_polynoms]).T


def get_interpolating_polynom(xs, ys):
    base_polynoms = [create_base_polynom(i) for i in range(len(xs))]
    A = create_polynom_linear_equation(xs, base_polynoms)
    merge_coeffs = gauss_elimination(A, ys)
    interpolating_polynom = merge_polynoms(base_polynoms, merge_coeffs)
    return interpolating_polynom
