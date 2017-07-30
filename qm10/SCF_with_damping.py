import numpy as np
from helper1 import *

np.set_printoptions(suppress=True, precision=4)


nbf, V, T, S, g, A, nuc, psi4_energy = buildInts("aug-cc-pvDZ",
"""
O
H 1 1.1
H 1 1.1 2 104
""")


e_conv = 1.e-6
d_conv = 1.e-6
nel = 5
damp_value = 0.20
damp_start = 5

# Core Hamiltonian
H = T + V

# print(S.shape)
# print(I.shape)

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
    E_total, E_diff = buildE(F, H, D, E_old, nuc)
    E_old = E_total
    print("Iter=%3d  E = % 16.12f  E_diff = % 8.4e  D_diff = % 8.4e" %
            (iteration, E_total, E_diff, grad_rms))

    # Break if e_conv and d_conv are met
    if (E_diff < e_conv) and (grad_rms < d_conv):
        break

    D = updateD(F, A, nel)

print("SCF has finished!\n")

print("Energy matches Psi4 %s" % np.allclose(psi4_energy, E_total))
