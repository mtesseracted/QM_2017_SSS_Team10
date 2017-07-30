from qm10 import helper2

"""
Hartree-Fock functions
"""
import psi4
import numpy as np


def hartree_fock(basis, mol, nel):
    """
    Returns:
    Hartree-Fock energy (float)

    Parameters:
    basis (multi-line string)
    geom (string) 
    """

    V, T, S, g, A = integrals(basis, mol)

    # Core Hamiltonian
    H = T + V

    A = a_funct(A)
    
    eps, C = diag(H, A)

    D = density_builder(C, nel)

    E_old = 0.0
    F_old = None
    for iteration in range(25):
        # F_pq = H_pq + 2 * g_pqrs D_rs - g_prqs D_rs
    
        # g = (7, 7, 7, 7)
        # D = (1, 1, 7, 7)
        # Jsum = np.sum(g * D, axis=(2, 3))
        J = np.einsum("pqrs,rs->pq", g, D)
        K = np.einsum("prqs,rs->pq", g, D)
    
        F_new = H + 2.0 * J - K
    
        #### Parameters F_old, F_new, and iteration;
           # returns F
        # conditional iteration > start_damp
        F = helper2.damping_func(iteration, 5, F_old, F_new, 0.2)
        
        F_old = F_new
    
        grad_rms = helper2.gradient(F, D, S)
    
        HF_energy = helper2.energy_conv(F, H, D, E_old, mol)

        eps, C = diag(F, A)
        Cocc = C[:, :nel]
        D = Cocc @ Cocc.T

        return(HF_energy)
  

def integrals(basis, mol):
    """
    Returns:
    K - kinetic integrals (Numpy array)
    S - overlap integrals (Numpy array)
    V - potential energy (Numpy array)
    A - electron repulsion integrals
        (Psi4 molecular integral object)

    Parameters:
    basis (multi-line string)
    mol (psi4.core.Molecule) 
    """

    # geom
    #mol = psi4.geometry(geom)
    
    # Build a molecule
    #mol.update_geometry()
    #mol.print_out()
    
    # Build a basis
    bas = psi4.core.BasisSet.build(mol, target="aug-cc-pVDZ")
    bas.print_out()
    
    # Build a MintsHelper
    mints = psi4.core.MintsHelper(bas)
    nbf = mints.nbf()

    ###check exception ###
    if (nbf > 100):
        raise Exception("More than 100 basis functions!")
    
    # Get integrals
    V = np.array(mints.ao_potential())
    T = np.array(mints.ao_kinetic())
    S = np.array(mints.ao_overlap())
    g = np.array(mints.ao_eri())
    A = mints.ao_overlap()

    return [V, T, S, g, A]

    


def a_funct(A):
    """
    Returns matrix raised to the -1/2 power 
        as a numpy array
    @A: psi4.core.Matrix
    """ 

    A.power(-0.5, 1.e-14)
    A = np.array(A)
    
    return A


def diag(F, A):
    """
    Returns eigenvalues and eigenvectors of matrix F
    @F: numpy.ndarray
    @A: numpy.ndarray
    """

    Fp = A.T @ F @ A
    #Fp = A.triplet(A, F, A, transa=True)
    eps, Cp = np.linalg.eigh(Fp)
    C = A @ Cp
    return eps, C


def density_builder(C, nel):
    """
    Returns the eigenvalues and eigenvectors of
    the core Hamiltonian
    """

    Cocc = C[:, :nel]
    D = Cocc @ Cocc.T
    return D 
