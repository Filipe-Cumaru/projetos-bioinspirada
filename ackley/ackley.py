import numpy as np

def ackley_function(x):
    """
    Implmentation of the Ackley function.
    """
    n = x.size
    c1, c2, c3 = 20, 0.2, 2*np.pi
    A = -c2*np.sqrt(np.sum(x**2) / n)
    B = np.sum(np.cos(c3*x)) / n
    f = -c1*np.exp(A) - np.exp(B) + c1 + np.e
    return f