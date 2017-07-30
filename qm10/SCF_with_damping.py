import numpy as np
import psi4
from helper1 import *

np.set_printoptions(suppress=True, precision=4)

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")

# Build a molecule
mol.update_geometry()
mol.print_out()

e_conv = 1.e-6
d_conv = 1.e-6
nel = 5
damp_value = 0.20
damp_start = 5

# Build a basis
bas = psi4.core.BasisSet.build(mol, target="aug-cc-pVDZ")
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

D = updateD(H, A, nel)

E_old = 0.0
F_old = np.zeros((nbf,nbf))
for iteration in range(25):
    # F_pq = H_pq + 2 * g_pqrs D_rs - g_prqs D_rs

    # g = (7, 7, 7, 7)
    # D = (1, 1, 7, 7)
    # Jsum = np.sum(g * D, axis=(2, 3))
    if iteration >= damp_start:
        F_old, F = buildF(g, D, H, damp_value, F_old)
    else:
        F_old, F = buildF(g, D, H, 0.0, F_old)

    # Build the AO gradient
    grad_rms = buildgrad(F, D, S)

    # Build the energy
    E_total, E_diff = buildE(F, H, D, E_old, mol.nuclear_repulsion_energy())
    E_old = E_total
    print("Iter=%3d  E = % 16.12f  E_diff = % 8.4e  D_diff = % 8.4e" %
            (iteration, E_total, E_diff, grad_rms))

    # Break if e_conv and d_conv are met
    if (E_diff < e_conv) and (grad_rms < d_conv):
        break

    D = updateD(F, A, nel)

print("SCF has finished!\n")

psi4.set_output_file("output.dat")
psi4.set_options({"scf_type": "pk"})
psi4_energy = psi4.energy("SCF/aug-cc-pVDZ", molecule=mol)
print("Energy matches Psi4 %s" % np.allclose(psi4_energy, E_total))
