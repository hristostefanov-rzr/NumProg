from misc import Polynomial


# Newton's method for finding roots of a polynomial
# Takes as an input a polynomial and returns a root
def newton_method(p, x0, k=1, eps=1e-5, max_iter=100):
    p_prim = get_derivative(p.coeff)
    x = x0
    for i in range(max_iter):
        x = x - k * (p(x) / p_prim(x))
        if abs(p(x)) < eps:
            return x
    return x


def get_derivative(p):
    return Polynomial([p[i] * i for i in range(1, len(p))])
