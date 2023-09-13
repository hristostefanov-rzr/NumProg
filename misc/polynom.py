import numpy as np


class Polynom:
    def __init__(self, coeff):
        self.coeff = np.array(coeff)

    def evaluate(self, x):
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
