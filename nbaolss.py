#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def linGL1(size):
    lower_diag = np.ones(size)
    diag = -2 * np.ones(size)
    upper_diag = np.ones(size)

    diag_values = [lower_diag, diag, upper_diag]
    diags = [-1, 0, 1]
    
    return (size + 1)**2 * scipy.sparse.spdiags(diag_values, diags, m=size,
                                          n=size, format="csr")
def fGL1(u, size, mu):
    return linGL1(size) @ u  + mu * (u  - u ** 3 / 3) 

def JacGL1(u, size, mu):
    nonlinear_part = mu * (1 - u ** 2)
    return linGL1(size) + scipy.sparse.diags(nonlinear_part, 0, format="csr")

# Function that computes Newton-Rhapson iterations
# to find a zero of a function
def newton(x0, func, jac_func, tol, max_iter):
    iter = 1
    norm_dx = 2 * tol
    x = x0
    while iter < max_iter and norm_dx > tol:
        jac = jac_func(x)
        dx = - spsolve(jac_func(x), func(x))
        x += np.real(dx)
        norm_dx = np.linalg.norm(dx)
        iter = iter + 1
    print('Newton converged after ' + str(iter) + ' iterations.')
    return x
        
def zero_of_GL1(u0, size, mu):
    tol = 1e-6
    max_iter = 200
    l_fGL1 = lambda u: fGL1(u, size, mu)
    l_JacGL1 = lambda u: JacGL1(u, size, mu)
    return newton(u0, l_fGL1, l_JacGL1, tol, max_iter)

