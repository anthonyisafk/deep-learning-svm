"""
Kernel function implementations. Not layed out with an Enum class,
like the activation functions built for MLP's.
The older design didn't feel modular enough.
================================================================
See https://github.com/anthonyisafk/deep-learning-mlp/blob/master/src/activation.py
"""

import numpy as np

class Kernel:
    f:callable # selected kernel function

    def __init__(self,
        f:str,
        tau=0, d=1,  # polynomial
        sigma=1,        # rbf
        k=1, theta=0 # tanh
    ):
        if f == "linear":
            self.f = linear()
        elif f == "poly":
            self.f = poly(tau, d)
        elif f == "rbf":
            self.f = rbf(sigma)
        elif f == "tanh":
            self.f = tanh(k, theta)
        else:
            error = f"Unknown kernel function : {f}"
            raise Exception(error)


    def __call__(self, x1:np.ndarray, x2:np.ndarray):
        return self.f(x1, x2)


def linear():
    def impl(x1:np.ndarray, x2:np.ndarray):
        return np.dot(x1, x2)
    return impl

def poly(tau, d):
    def impl(x1:np.ndarray, x2:np.ndarray):
        return (np.dot(x1, x2) + tau) ** d
    return impl

def rbf(sigma):
    sigma_sqr = sigma ** 2
    def impl(x1:np.ndarray, x2:np.ndarray):
        diff = x1 - x2
        norm_diff = np.linalg.norm(diff) ** 2
        return np.exp(-norm_diff / sigma_sqr)
    return impl

def tanh(k, theta):
    def impl(x1:np.ndarray, x2:np.ndarray):
        return np.tanh(k * np.dot(x1, x2) + theta)
    return impl

