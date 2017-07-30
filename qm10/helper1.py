"""
Hartree-Fock functions
"""
import psi4
import numpy as np

# Diagonalize Core H
def diag(F, A):
    Fp = A.T @ F @ A
    eps, Cp = np.linalg.eigh(Fp)
    C = A @ Cp
    return eps, C


def updateD(H, A, nel):
    eps, C = diag(H, A)
    Cocc = C[:, :nel]
    D = Cocc @ Cocc.T
    return D

def buildF(g, D, H, dampVal, F_old):
    J = np.einsum("pqrs,rs->pq", g, D)
    K = np.einsum("prqs,rs->pq", g, D)

    F_new = H + 2.0 * J - K

    # conditional iteration > start_damp
    F = dampVal * F_old + (1.0 - dampVal) * F_new

    F_old = F_new
    # F = (damp_value) Fold + (??) Fnew
    return F_old, F

def buildgrad(F, D, S):
    grad = F @ D @ S - S @ D @ F

    grad_rms = np.mean(grad ** 2) ** 0.5

    return grad_rms

def buildE(F, H, D, old, nuc):
    # here, nuc is mol.nuclear_repulsion_energy()
    elec = np.sum((F + H) * D)
    total = elec + nuc

    diff = total - old

    return total, diff
    


