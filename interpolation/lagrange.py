import numpy as np

from misc import FactorisedPolynom, merge_polynoms


class LagrangePolynoms:
    def __init__(self, xs, ys):
        self.lagrange_polynoms = self.compute_lagrange_polynoms(xs)
        self.ys = ys
        self.standard_polynom = self.calculate_standard_polynom()

    def compute_lagrange_polynoms(self, xs):
        lagrange_polynoms = []
        for i in range(len(xs)):
            xs_without_i = xs[i != np.arange(len(xs))]
            coeff = 1 / (np.prod(xs[i] - xs_without_i))
            lagrange_polynoms.append(FactorisedPolynom(xs_without_i, coeff))
        return lagrange_polynoms

    def calculate_standard_polynom(self):
        polynoms = [
            self.lagrange_polynoms[i].to_standard_polynom()
            for i in range(len(self.lagrange_polynoms))
        ]
        return merge_polynoms(polynoms, self.ys)

    def to_standard_polynom(self):
        return self.standard_polynom

    def __call__(self, x):
        return self.standard_polynom(x)

    def __str__(self):
        functs = "\n".join(
            [
                p.to_named_string(function_name=f"L{i}(x)")
                for i, p in enumerate(self.lagrange_polynoms)
            ]
        )
        return (
            functs
            + "\nf(x) = "
            + " + ".join([f"({y}) * L{i}(x)" for i, y in enumerate(self.ys)])
        )

    def __repr__(self):
        return str(self)
