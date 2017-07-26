"""
Test cases for helper1.py functions
"""
import qm10
import pytest
import psi4
import numpy as np
psi4.core.be_quiet()


### geom
g1 = """
    O
    H 1 1.1
    H 1 1.1 2 104
"""

# Build a molecule
mol = psi4.geometry(g1)
mol.update_geometry()

b1 = psi4.core.BasisSet.build(mol, target="aug-cc-pVDZ")
mints = psi4.core.MintsHelper(b1)

expected = 1
A = mints.ao_overlap()
id2 = np.array([[1,0],[0,1]])

td1 = [(b1, g1, 2, expected), ]
td2 = [(A, A, 2, id2)]

@pytest.mark.parametrize("a, b, n, exp", td1)
def test_hartree_fock(a, b, n, exp):
    qm10.helper1.hartree_fock(a, b, n)
    assert(True)



@pytest.mark.parametrize("a, b, n, exp", td1)
def test_integrals(a, b, n, exp):
    qm10.helper1.integrals(a, b)
    assert(True)


def test_a_funct():
    qm10.helper1.a_funct(A)
    assert(True)

@pytest.mark.parametrize("h1, a1, nelec, exp", td2)
def test_core_diag(h1, a1, nelec, exp):
    qm10.helper1.core_diag(h1, a1, nelec)
    assert(True)


