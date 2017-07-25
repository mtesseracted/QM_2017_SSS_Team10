import numpy
import psi4

np.set_printoptions(suppress=True, precision=4)

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")

# Build a molecule
mol.update_geometry()
mol.print_out()

nel = 5

# Build a basis
bas = psi4.core.BasisSet.build(mol, target="sto-3g")
bas.print_out()

# Build a MintsHelper
mints = psi4.core.MintsHelper(bas)
nbf = mints.nbf()

if (nbf > 100):
    raise Exception("More than 100 basis functions!")

V = np.array(mints.ao_potential())
T = np.array(mints.ao_kinetic())

# Core Hamiltonian
H = T + V

S = np.array(mints.ao_overlap())
g = np.array(mints.ao_eri())

# print(S.shape)
# print(I.shape)

A = mints.ao_overlap()
A.power(-0.5, 1.e-14)
A = np.array(A)

# print(A @ S @ A)

# Diagonalize Core H
Fp = A.T @ H @ A
eps, Cp = np.linalg.eigh(Fp)
C = A @ Cp

Cocc = C[:, :nel]
D = Cocc @ Cocc.T

# F_pq = H_pq + 2 * g_pqrs D_rs - g_prqs D_rs

# g = (7, 7, 7, 7)
# D = (1, 1, 7, 7)
#Jsum = np.sum(g * D, axis=(2, 3))
J = np.einsum("pqrs,rs->pq", g, D)
K = np.einsum("prqs,rs->pq", g, D)

F = H + 2.0 * J - K

print(F)