import numpy as np
from misc import FactorisedPolynomial, merge_polynomials


class NewtonTable:
    def __init__(self, xs, ys):
        # xs must be sorted
        self.xs = xs
        self.cs = np.zeros((len(xs), len(xs)))
        self.cs[:, 0] = ys
        for k in range(1, len(xs)):
            for i in range(len(xs) - k):
                self.cs[i][k] = (self.cs[i + 1][k - 1] - self.cs[i][k - 1]) / (
                    self.xs[i + k] - self.xs[i]
                )

        self.polynomial = self.calculate_polynomial()

    def add_points_after(self, new_xs, new_ys):
        # new_xs must be sorted
        cs = self.cs
        n = len(new_xs) + len(cs)
        self.xs = np.concatenate((self.xs, new_xs))
        new_cs = np.zeros((n, n))
        new_cs[: len(cs), : len(cs)] = cs
        new_cs[len(cs) :, 0] = new_ys
        for k in range(1, n):
            for i in range(max(len(cs) - k, 0), len(cs) + len(new_xs) - k):
                new_cs[i][k] = (new_cs[i + 1][k - 1] - new_cs[i][k - 1]) / (
                    self.xs[i + k] - self.xs[i]
                )
        self.cs = new_cs
        self.polynomial = self.calculate_polynomial()

    def add_points_before(self, new_xs, new_ys):
        # xs must be sorted
        cs = self.cs
        xs = np.concatenate((new_xs, self.xs))
        n = len(new_xs) + len(cs)
        new_cs = np.zeros((n, n))
        new_cs[len(new_xs) :, : len(cs)] = cs
        new_cs[: len(new_xs), 0] = new_ys
        for k in range(1, len(cs) + 1):
            for i in range(len(new_xs)):
                new_cs[i][k] = (new_cs[i + 1][k - 1] - new_cs[i][k - 1]) / (
                    xs[i + k] - xs[i]
                )
        start = 0
        for k in range(len(cs) + 1, n):
            start += 1
            for i in range(len(new_xs) - start):
                new_cs[i][k] = (new_cs[i + 1][k - 1] - new_cs[i][k - 1]) / (
                    xs[i + k] - xs[i]
                )
        self.cs = new_cs
        self.xs = xs
        self.polynomial = self.calculate_polynomial()

    def calculate_polynomial(self):
        polynomials = []
        for i in range(len(self.xs) - 1):
            total_coeff = self.cs[0][i + 1]
            p = FactorisedPolynomial(self.xs[: i + 1], total_coeff)
            polynomials.append(p.to_standard_polynomial())
        final_polynom = merge_polynomials(polynomials, np.ones(len(polynomials)))
        final_polynom.coeff[0] += self.cs[0][0]
        return final_polynom

    def __call__(self, x):
        return self.polynomial(x)
