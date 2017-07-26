import numpy as np
import psi4


def fock_diag(A, F):
    '''
    This functions diagonalizes the fock matrix
    '''
    pass


def damping_func(iteration, damp_start, F_old, F_new):
    '''
    This fuction updates the fock matrix by a damping fuction
    '''
    if iteration >= damp_start:
        F = damp_value * F_old + (1.0 - damp_value) * F_new
    else:
        F = F_new
    return F


def gradient(F, D, S):
   '''
   This functions returns the gradient and its root mean square 
   '''
   grad = F @ D @ S - S @ D @ F
   grad_rms = np.mean(grad ** 2) ** 0.5
   return grad, grad_rms


def energy_conv(F, H, D, E_old):
   '''
   This function builds the electronic energy from the Fock, density 
   and core hamiltonian matrices
   '''
   E_electric = np.sum((F + H) * D)
   E_total = E_electric + mol.nuclear_repulsion_energy()
   E_diff = E_total - E_old
   E_old = E_total
   return E_total, E_diff


