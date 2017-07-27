import numpy as np
import psi4

np.set_printoptions(suppress=True, precision=4)

##############################
# FUNCTIONS SPACE
#############################


def damping_func(iteration, damp_start, F_old, F_new, damp_value):
    if iteration >= damp_start:
        F = damp_value * F_old + (1.0 - damp_value) * F_new
    else:
        F = F_new
    print(type(damp_value), "<-damp_value")
    return F

def gradient(F, D, S):
    grad = F @ D @ S - S @ D @ F
    grad_rms = np.mean(grad ** 2) ** 0.5
    return grad, grad_rms

def energy_conv(F, H, D, E_old):
    E_electric = np.sum((F + H) * D)
    E_total = E_electric + mol.nuclear_repulsion_energy()

    E_diff = E_total - E_old
    E_old = E_total
    print(type(F),"<-F")
    print(type(H),"<-H")
    print(type(D),"<-D")
    print(type(E_old),"<-E_old")
    print(type(E_diff),"<-E_diff")
    return E_total, E_diff

def update_D(diag, F, A):
    '''
    This function updates the density matrix(D) for the next iteration
    
    Arguments |  Datatype      | description
              |                |  
    diag      | function       | Diagonalizes the core Hamiltonian
    F         | numpy array    | Fock matrix
    A         | numpy array    | Transformation matrix
    eps       | numpy array    | Eigenvalues of core Hamiltonian 
    D         | numpy array    | Density matrix 
    C         | numpy array    | Orthonormal eigenvectors of core Hamiltonian
    '''
    eps, C = diag(F, A)
    Cocc = C[:, :nel]
    D = Cocc @ Cocc.T
    return D, eps





### geom
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")

# Build a molecule
mol.update_geometry()
mol.print_out()
###
    
e_conv = 1.e-6
d_conv = 1.e-6
nel = 5
damp_value = 0.20
damp_start = 5

### Build a basis
bas = psi4.core.BasisSet.build(mol, target="aug-cc-pVDZ")
bas.print_out()
###
    
###Build a MintsHelper
mints = psi4.core.MintsHelper(bas)
nbf = mints.nbf()
###

###check exception ###
if (nbf > 100):
    raise Exception("More than 100 basis functions!")
###

### Core Hamiltonian
    
V = np.array(mints.ao_potential())
T = np.array(mints.ao_kinetic())

H = T + V
####


S = np.array(mints.ao_overlap())
g = np.array(mints.ao_eri())

# print(S.shape)
# print(I.shape)

A = mints.ao_overlap()
A.power(-0.5, 1.e-14)
A = np.array(A)

# print(A @ S @ A)


### Diagonalize Core H
def diag(F, A):
    Fp = A.T @ F @ A
    eps, Cp = np.linalg.eigh(Fp)
    C = A @ Cp
    return eps, C
####
    
eps, C = diag(H, A)
Cocc = C[:, :nel]
D = Cocc @ Cocc.T

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

    # conditional iteration > start_damp
    F = damping_func(iteration, damp_start, F_old, F_new, damp_value)
    F_old = F_new
    # F = (damp_value) Fold + (??) Fnew

    # Build the AO gradient
    grad, grad_rms = gradient(F, D, S)

    # Build the energy
    E_total, E_diff = energy_conv(F, H, D, E_old) 
    print("Iter=%3d  E = % 16.12f  E_diff = % 8.4e  D_diff = % 8.4e" %
            (iteration, E_total, E_diff, grad_rms))

    # Break if e_conv and d_conv are met
    if (E_diff < e_conv) and (grad_rms < d_conv):
        break

    D, eps = update_D(diag, F, A)

print("SCF has finished!\n")




psi4.set_output_file("output.dat")
psi4.set_options({"scf_type": "pk"})
psi4_energy = psi4.energy("SCF/aug-cc-pVDZ", molecule=mol)
print("Energy matches Psi4 %s" % np.allclose(psi4_energy, E_total))

# I can edit files
