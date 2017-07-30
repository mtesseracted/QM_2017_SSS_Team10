"""
Hartree-Fock functions
"""
import psi4
import numpy as np


def buildInts(basis, geom):
    mol = psi4.geometry(geom)
    # Build a molecule
    mol.update_geometry()
    mol.print_out()

    # Build a basis
    bas = psi4.core.BasisSet.build(mol, target=basis)
    bas.print_out()

    # Build a MintsHelper
    mints = psi4.core.MintsHelper(bas)
    nbf = mints.nbf()
    
    if (nbf > 100):
        raise Exception("More than 100 basis functions!")
    
    V = np.array(mints.ao_potential())
    T = np.array(mints.ao_kinetic())
    
    A = mints.ao_overlap()
    A.power(-0.5, 1.e-14)
    A = np.array(A)

    S = np.array(mints.ao_overlap())
    g = np.array(mints.ao_eri())
    
    nuc_e = mol.nuclear_repulsion_energy()

    psi4.set_output_file("output.dat")
    psi4.set_options({"scf_type": "pk"})
    psi4_e = psi4.energy("SCF/" + basis, molecule=mol)

    return nbf, V, T, S, g, A, nuc_e, psi4_e

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
    


