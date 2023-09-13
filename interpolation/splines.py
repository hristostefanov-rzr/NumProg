import numpy as np
from misc.polynom import Polynom

H0 = Polynom([1, 0, -3, 2])
H1 = Polynom([0, 0, 3, -2])
H2 = Polynom([0, 1, -2, 1])
H3 = Polynom([0, 0, -1, 1])


# Creates the tri matrix used to calculate the derivatives needed for the spline interpolation
def create_tri_matrix(n):
    smaller = np.ones(n - 1)
    diagonal = 4 * np.ones(n)
    return np.diag(smaller, -1) + np.diag(diagonal, 0) + np.diag(smaller, 1)


# Using the sampled points and the derivatives at the edges, this function returns the derivatives at the inner points
# If no edge derivatives are given, the function assumes that the second derivative is 0 at the edges
def find_derivatives(xs, ys, edge_derivatives=None):
    n = len(xs) - 1
    tri_matrix = create_tri_matrix(n - 1)
    h = (xs[-1] - xs[0]) / n
    rhs = (ys[2:] - ys[:-2]) * (3 / h)
    if edge_derivatives is None:
        edge_derivatives = [0, 0]
    known_derivatives = np.zeros(n - 1)
    known_derivatives[0] = edge_derivatives[0]
    known_derivatives[-1] = edge_derivatives[-1]
    rhs = rhs - known_derivatives
    inner_derivatives = np.linalg.solve(tri_matrix, rhs)
    inner_derivatives = np.pad(
        inner_derivatives, (1, 1), mode="constant", constant_values=0
    )
    inner_derivatives[0] = edge_derivatives[0]
    inner_derivatives[-1] = edge_derivatives[-1]
    return inner_derivatives


# Transform the input x which has to be in the interval [lower_bound, upper_bound] to the interval [0, 1]
class TransformFunction:
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.size = upper_bound - lower_bound

    def evaluate(self, x):
        return (x - self.lower_bound) / self.size

    def __call__(self, x):
        return self.evaluate(x)


# Uses the sampled points and the derivatives
# at the edges to create a spline
class Spline:
    def __init__(self, val_start, val_end, der_start, der_end, transform_function):
        self.val_start = val_start
        self.val_end = val_end
        self.der_start = der_start
        self.der_end = der_end
        self.transform_function = transform_function

    def evaluate(self, x):
        t_x = self.transform_function(x)
        h = self.transform_function.size
        result = self.val_start * H0(t_x)
        result += self.val_end * H1(t_x)
        result += h * self.der_start * H2(t_x)
        result += h * self.der_end * H3(t_x)
        return result

    def __call__(self, x):
        return self.evaluate(x)


# A class that combines the splines into a function that can be evaluated
class HermiteFunction:
    def __init__(self, splines, xs):
        self.splines = splines
        self.xs = xs

    def evaluate(self, x):
        spline_to_use = np.searchsorted(self.xs, x) - 1
        if spline_to_use is None or spline_to_use < 0:
            spline_to_use = 0
        if spline_to_use >= len(self.splines):
            spline_to_use = len(self.splines) - 1
        return self.splines[spline_to_use](x)

    def __call__(self, x):
        return self.evaluate(x)


# Creates the interpolating function from the sampled points and the derivatives at the edges
def create_hermite_function(xs, ys, derivatives):
    splines = []
    for i in range(len(xs) - 1):
        transform_function = TransformFunction(xs[i], xs[i + 1])
        spline = Spline(
            ys[i], ys[i + 1], derivatives[i], derivatives[i + 1], transform_function
        )
        splines.append(spline)
    return HermiteFunction(splines, xs)


# Takes as input two numpy arrays of the same size
# xs contains the x values of the sampled points
# ys contains the y values of the sampled points
# edge_derivatives is a list of the derivatives at the edges
# if no edge_derivatives are given, the function assumes that the second derivative is 0 at the edges
# returns a function that can be evaluated at any point in the interval [xs[0], xs[-1]]
def interpolate_with_splines(xs, ys, edge_derivatives=None):
    derivatives = find_derivatives(xs, ys, edge_derivatives)
    return create_hermite_function(xs, ys, derivatives)
