import numpy as np


def create_base_polynom(degree):
    degree = degree
    coeffs = [0] * (degree + 1)
    coeffs[-1] = 1
    base_polynom = Polynom(coeffs)
    return base_polynom


def merge_polynoms(polynoms, merge_coeffs):
    assert len(polynoms) == len(
        merge_coeffs
    ), "The number of polynoms and coefficients must be the same"
    polynom_coeff = [polynoms[i].coeff for i in range(len(polynoms))]
    max_degree = max([len(coeff) for coeff in polynom_coeff])
    for i in range(len(polynom_coeff)):
        polynom_coeff[i] = np.pad(
            polynom_coeff[i], (0, max_degree - len(polynom_coeff[i]))
        )
    polynom_coeff = np.array(polynom_coeff)
    merge_coeffs = np.array(merge_coeffs)
    result = np.sum(polynom_coeff * merge_coeffs[:, None], axis=0)
    return Polynom(result)


class Polynom:
    def __init__(self, coeff):
        self.coeff = np.array(coeff)

    def evaluate(self, x):
        if isinstance(x, np.ndarray):
            x = x[:, None]
        xs = np.power(x, np.arange(len(self.coeff)))
        return xs @ self.coeff

    def __call__(self, x):
        return self.evaluate(x)

    def __str__(
        self,
    ):
        terms = []
        first_written = False
        for i, coef in enumerate(self.coeff):
            if coef == 0:
                continue

            term = f"{abs(coef)}"
            if i == 0:
                term = f"-{term}"
                terms.append(term)
                first_written = True
                continue

            if coef != 1 and coef != -1:
                term += "x"
            else:
                term = "x"
            if i != 1:
                term += f"^{i}"
            if first_written:
                if coef < 0:
                    term = f" - {term}"
                else:
                    term = f" + {term}"
            else:
                first_written = True
                if coef < 0:
                    term = f" -{term}"
            terms.append(term)

        return "f(x) = " + "".join(terms)

    def __repr__(
        self,
    ):
        return "Polynomial: " + self.__str__()


# This class represents a factorised polynomial total_coeff * (x-a)(x-b)(x-c)...
class FactorisedPolynom:
    def __init__(self, roots, total_coeff=1.0):
        self.roots = np.array(roots)
        self.total_coeff = total_coeff

    def evaluate(self, x):
        return self.total_coeff * np.prod(x - self.roots)

    def __call__(self, x):
        return self.evaluate(x)

    def to_named_string(self, function_name="f(x)"):
        factored_terms = []
        for i in range(len(self.roots)):
            if self.roots[i] == 0:
                factored_terms.append("x")
            elif self.roots[i] > 0:
                factored_terms.append("(x - " + str(self.roots[i]) + ")")
            else:
                factored_terms.append("(x + " + str(-self.roots[i]) + ")")
        return (
            f"{function_name} = "
            + str(self.total_coeff)
            + " * "
            + " * ".join(factored_terms)
        )

    def __str__(self):
        return self.to_named_string("f(x)")

    def __repr__(
        self,
    ):
        return "Factorised polynomial: " + self.__str__()

    def to_standard_polynom(self):
        coefficients = np.array([self.total_coeff])
        for i in range(len(self.roots)):
            coefficients = self.multiply_by_factor(coefficients, self.roots[i])
        return Polynom(coefficients)

    def multiply_by_factor(self, coeff, root):
        new_coeff = np.append(coeff, 0) * -root
        new_coeff[1:] += coeff
        return new_coeff
