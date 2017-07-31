import numpy as np
import psi4

# Make sure we get the same random array
np.random.seed(0)

# A hydrogen molecule
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")

mol.update_geometry()

# Build a ERI tensor
basis = psi4.core.BasisSet.build(mol, target="cc-pVDZ")
mints = psi4.core.MintsHelper(basis)
I = np.array(mints.ao_eri())


# Symmetric random density
nbf = I.shape[0]
D = np.random.rand(nbf, nbf)
D = (D + D.T) / 2

# Reference
J_ref = np.einsum("pqrs,rs->pq", I, D)
K_ref = np.einsum("prqs,rs->pq", I, D)

# Your implementation
J = np.random.rand(nbf, nbf)
K = np.random.rand(nbf, nbf)

# Make sure your implementation is correct
print("J is correct: %s" % np.allclose(J, J_ref))
print("K is correct: %s" % np.allclose(K, K_ref))

wfn = psi4.core.Wavefunction.build(mol,basis)

# Get orbital basis from a Wavefunction object
orb = wfn.basisset()
# Build the complementary JKFIT basis for the aug-cc-pVDZ basis (for example)
aux = psi4.core.BasisSet.build(mol, fitrole="JKFIT", other="aug-cc-pVDZ")
# The zero basis set
zero_bas = psi4.core.BasisSet.zero_ao_basis_set()
# Build instance of MintsHelper
mints = psi4.core.MintsHelper(orb)


# Build (P|pq) raw 3-index ERIs, dimension (1, Naux, nbf, nbf)
Qls_tilde = mints.ao_eri(zero_bas, aux, orb, orb)
Qls_tilde = np.squeeze(Qls_tilde) # remove the 1-dimensions
# Build & invert Coulomb metric, dimension (1, Naux, 1, Naux)
metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
metric.power(-0.5, 1.e-14)
metric = np.squeeze(metric) # remove the 1-dimensions

Qls = np.einsum('PQ,Qls->Pls', metric, Qls_tilde)


# Coulomb matrix
chiP = np.einsum('Pls,ls->P', Qls, D)
Jp = np.einsum('Pmn,P->mn', Qls, chiP)

print("J is correct: %s" % np.allclose(Jp, J_ref))

print(Jp-J_ref)


print(Jp.min())

print(J_ref.min())


# Exchange matrix

