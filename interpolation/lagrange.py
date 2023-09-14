import numpy as np

from misc import FactorisedPolynomial, merge_polynomials


def compute_lagrange_polynomials(xs):
    lagrange_polynomials = []
    for i in range(len(xs)):
        xs_without_i = xs[i != np.arange(len(xs))]
        coeff = 1 / (np.prod(xs[i] - xs_without_i))
        lagrange_polynomials.append(FactorisedPolynomial(xs_without_i, coeff))
    return lagrange_polynomials


class LagrangePolynomials:
    def __init__(self, xs, ys):
        self.lagrange_polynomials = compute_lagrange_polynomials(xs)
        self.ys = ys
        self.standard_polynomial = self.calculate_standard_polynomial()

    def calculate_standard_polynomial(self):
        polynomials = [
            self.lagrange_polynomials[i].to_standard_polynomial()
            for i in range(len(self.lagrange_polynomials))
        ]
        return merge_polynomials(polynomials, self.ys)

    def to_standard_polynomial(self):
        return self.standard_polynomial

    def __call__(self, x):
        return self.standard_polynomial(x)

    def __str__(self):
        functs = "\n".join(
            [
                p.to_named_string(function_name=f"L{i}(x)")
                for i, p in enumerate(self.lagrange_polynomials)
            ]
        )
        return (
            functs
            + "\nf(x) = "
            + " + ".join([f"({y}) * L{i}(x)" for i, y in enumerate(self.ys)])
        )

    def __repr__(self):
        return str(self)
