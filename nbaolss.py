#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import eig
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
    #print('Newton converged after ' + str(iter) + ' iterations.')
    return x
        
def zero_of_GL1(u0, size, mu):
    tol = 1e-6
    max_iter = 200
    l_fGL1 = lambda u: fGL1(u, size, mu)
    l_JacGL1 = lambda u: JacGL1(u, size, mu)
    return newton(u0, l_fGL1, l_JacGL1, tol, max_iter)

# Sorts the eigenvalues. from the smallest in magnitude to the biggest
# Also rearranges the eigenvectors in the proper way
def sort(eig_val, eig_vec):
    # Calculate the new indices
    idx = np.argsort(np.abs(eig_val))
    
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]
    return eig_val, eig_vec

def epsilon(gamma):
    return 2 * np.sqrt(1 - 1 / (1 + gamma))

# Finds initial conditions on num_branches branches
def init(num_branches, size):
    gamma = 0.1
    eps = epsilon(gamma)
    
    eig_val, eig_vec = eig(-linGL1(size).A)
    eig_val, eig_vec = sort(eig_val, eig_vec)
    
    res = np.zeros((num_branches, size))
    mus = np.zeros((num_branches,))
    for idx in range(0, num_branches):
        uk = eps * eig_vec[:, idx] / np.max(eig_vec[:, idx])
        mus[idx] = np.real((1 + gamma) * eig_val[idx])
        res[idx, :] = zero_of_GL1(uk, size, mus[idx])
    return res, mus
    
def cont(mu, u0, mu_end, steps):
    size = len(u0)
    mu_steps = np.linspace(mu, mu_end, steps)
    res = np.zeros((steps, size))
    res[0, :] = zero_of_GL1(u0, size, mu_steps[0])
    for step in range(1, steps):
        res[step, :] = zero_of_GL1(res[step - 1], size, mu_steps[step])
    return res
