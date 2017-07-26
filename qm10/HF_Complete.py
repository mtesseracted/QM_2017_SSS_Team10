import numpy as np
import psi4

np.set_printoptions(suppress=True, precision=4)

e_conv = 1.e-6
d_conv = 1.e-6
nel = 5
damp_value = 0.20
damp_start = 5

HF_energy = hartree_fock("aug-cc-pVDZ",
"""
O
H 1 1.1
H 1 1.1 2 104
""")

print("SCF has finished!\n")

print(HF_energy)

# Compare to Psi4
psi4.set_output_file("output.dat")
psi4.set_options({"scf_type": "pk"})
psi4_energy = psi4.energy("SCF/aug-cc-pVDZ", molecule=mol)
print("Energy matches Psi4 %s" % np.allclose(psi4_energy, HF_energy))


##############################
def hartree_fock(basis, geom):
    """
    Function that returns the Hartree-Fock
    energy of a molecule with basis set basis
    and geometry geom
    """

    V, T, S, g, A = integrals(basis, geom)

    # Core Hamiltonian
    H = T + V

#### A_funct
    A.power(-0.5, 1.e-14)
    A = np.array(A)
####

#### Could be a function that accepts H, A, nel
   # and outputs D
    # Diagonalize core Hamiltonian
    eps, C = core_diag(H, A)
    Cocc = C[:, :nel]
    D = Cocc @ Cocc.T
####

####
    # Diagonalize Fock matrix
    Fp = A.T @ F @ A
    eps, Cp = np.linalg.eigh(Fp)
    C = A @ Cp
    return eps, C
####

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
        if iteration >= damp_start:
            F = damp_value * F_old + (1.0 - damp_value) * F_new
        else:
            F = F_new
        ####
        
        F_old = F_new
    
        #### parameters F, D, S; returns grad_rms
        # Build the AO gradient
        grad = F @ D @ S - S @ D @ F
    
        grad_rms = np.mean(grad ** 2) ** 0.5
        ####
    
        ####
        # Build the energy
        E_electric = np.sum((F + H) * D)
        E_total = E_electric + mol.nuclear_repulsion_energy()
    
        E_diff = E_total - E_old
        E_old = E_total
        print("Iter=%3d  E = % 16.12f  E_diff = % 8.4e  D_diff = % 8.4e" %
                (iteration, E_total, E_diff, grad_rms))

        # Break if e_conv and d_conv are met
        if (E_diff < e_conv) and (grad_rms < d_conv):
            break
        ####
    
        #### Function - but we already have a function
           # like this
        eps, C = diag(F, A)
        Cocc = C[:, :nel]
        D = Cocc @ Cocc.T
        ####




def integrals(basis, geom):
    """
    Returns the kinetic, overlap, potential,
    and electron repulsion integrals of a
    molecule with basis set basis and
    geometry geom.
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



def core_diag(F, A):
    """
    Given a core Hamiltonian F and matrix A,
    returns the eigenvalues and eigenvectors
    of the core Hamiltonian
    """
    Fp = A.T @ F @ A
    eps, Cp = np.linalg.eigh(Fp)
    C = A @ Cp
    return eps, C
