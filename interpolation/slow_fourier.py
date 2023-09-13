import numpy as np


# Functions useful for the computation of the discrete fourier transform
def w(n):
    return np.exp((1j * 2 * np.pi) / n)


def w_ij(n):
    w_base = w(n)

    def value_ij(i, j):
        return w_base ** (i * j)

    return value_ij


# Takes as input a numpy array of data samples (sampled at equidistant points)
# Returns a numpy array of the same size with the fourier transform of the data samples
def slow_fourier_transform(data_samples):
    n = len(data_samples)
    creator_function = w_ij(n)
    transformer_matrix = np.fromfunction(creator_function, (n, n))
    return (1 / n) * (transformer_matrix @ data_samples)


# Takes as input the frequency domain data returned by the fourier transform
# Returns a function that can be evaluated at any point which goes through the sampled points
def construct_function(frequency_domain):
    def f(x):
        result = 0
        for i, weight in enumerate(frequency_domain):
            result += weight * (np.cos(i * x) - np.sin(i * x) * 1j)
        return result

    return f
