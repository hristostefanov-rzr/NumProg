import numpy as np

from misc.polynomial import create_base_polynomial, merge_polynomials
from lineq.gauss import gauss_elimination


# Takes as input the points to be interpolated
# uses a basis of polynomials to construct the interpolating polynomial
def create_polynom_linear_equation(xs, base_polynomials):
    return np.array([base_polynomial(xs) for base_polynomial in base_polynomials]).T

# Takes as input the points to be interpolated
# and calculates the interpolating polynomial
# using a basis of polynomials (1, x, x^2, ...)
def get_interpolating_polynomial(xs, ys):
    base_polynomials = [create_base_polynomial(i) for i in range(len(xs))]
    A = create_polynom_linear_equation(xs, base_polynomials)
    merge_coeffs = gauss_elimination(A, ys)
    interpolating_polynomial = merge_polynomials(base_polynomials, merge_coeffs)
    return interpolating_polynomial
