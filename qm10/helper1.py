"""
Hartree-Fock functions
"""

def hartree_fock(basis, geom):
    """
    Returns Hartree-Fock energy
    given basis set and geom
    """

    pass
  

def integrals(basis, geom):
    """
    Returns the kinetic, overlap, potential, and electron
    repulsion integrals of a molecule with basis set basis
    and geometry geom.
    """

    # geom
    mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    """)
    
    # Build a molecule
    mol.update_geometry()
    mol.print_out()
    
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
    Returns matrix raised to the -1/2
    power
    """ 

    A.power(-0.5, 1.e-14)
    A = np.array(A)


def core_diag(H, A, nel):
    """
    Returns the eigenvalues and eigenvectors of
    the core Hamiltonian
    """

    pass
    
