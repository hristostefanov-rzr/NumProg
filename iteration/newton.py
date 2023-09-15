from misc import Polynomial


# Newton's method for finding roots of a polynomial
# Takes as an input a polynomial and returns a root
def newton_method_polynomial(p, x0, k=1, eps=1e-5, max_iter=100):
    p_prim = get_derivative_polynomial(p.coeff)
    return newton_method(p, p_prim, x0, k, eps, max_iter)


# Newton's method for finding roots of a function when the derivative is known
def newton_method(f, f_prim, x0, k=1, eps=1e-5, max_iter=100):
    x = x0
    for i in range(max_iter):
        x = x - k * (f(x) / f_prim(x))
        if abs(f(x)) < eps:
            return x
    return x


# Takes as an input a polynomial and returns its derivative polynomial
def get_derivative_polynomial(p):
    return Polynomial([p[i] * i for i in range(1, len(p))])

# Secant method for finding roots of a function without knowing the derivative
def secant_method(f, x0, x1, eps=1e-5, max_iter=100):
    x = x1
    x_prev = x0
    for i in range(max_iter):
        x, x_prev = x - (f(x) * (x - x_prev)) / (f(x) - f(x_prev)), x
        if abs(f(x)) < eps:
            return x
    return x
