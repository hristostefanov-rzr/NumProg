import numpy as np


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
