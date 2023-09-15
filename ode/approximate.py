from iteration.newton import secant_method


# Implementations of approximate methods for solving ODEs
def eulers_method(t0, y0, f, delta_t, target_t):
    y = y0
    t = t0
    steps = int((target_t - t0) / delta_t)
    for i in range(steps):
        t = t + delta_t
        y = y + delta_t * f(t, y)
    return y


def heun_method(t0, y0, f, delta_t, target_t):
    y = y0
    t = t0
    steps = int((target_t - t0) / delta_t)
    for i in range(steps):
        t = t + delta_t
        T1 = y + delta_t * f(t, y)
        y = y + (delta_t / 2) * (f(t, y) + f(t + delta_t, T1))
    return y


def runge_kutta_method(t0, y0, f, delta_t, target_t):
    y = y0
    t = t0
    steps = int((target_t - t0) / delta_t)
    for i in range(steps):
        t = t + delta_t
        T1 = f(t, y)
        T2 = f(t + (delta_t / 2), y + (delta_t / 2) * T1)
        T3 = f(t + (delta_t / 2), y + (delta_t / 2) * T2)
        T4 = f(t + delta_t, y + delta_t * T3)

        y = y + (delta_t / 6) * (T1 + 2 * T2 + 2 * T3 + T4)
    return y


# A somewhat buggy implementation of the implicit Euler's method


# Implicit Euler that uses secant method to solve the equation
def implicit_euler_method(t0, y0, f, delta_t, target_t):
    steps = int((target_t - t0) / delta_t)
    y = y0
    t = t0
    for i in range(steps):
        t = t + delta_t

        def function_to_solve(next_y):
            return y + delta_t * f(t, next_y) - next_y

        y = secant_method(function_to_solve, y - 1, y + 1)
    return y
