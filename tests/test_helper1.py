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
m1 = psi4.geometry(g1)
m1.update_geometry()

b1 = psi4.core.BasisSet.build(m1, target="aug-cc-pVDZ")
mints = psi4.core.MintsHelper(b1)

expected = 1
A = mints.ao_overlap()
id2 = np.array([[1,0],[0,1]])

td1 = [(b1, g1, 2, expected), ]
td1b = [(b1, m1, 2, expected), ]
td2 = [(A, A, id2)]


@pytest.mark.parametrize("h1, a1, exp", td2)
def test_diag(h1, a1, exp):
    qm10.helper1.diag(h1, np.array(a1))
    assert(True)


@pytest.mark.parametrize("a, b, n, exp", td1b)
def test_hartree_fock(a, b, n, exp):
    qm10.helper1.hartree_fock(a, b, n)
    assert(True)



@pytest.mark.parametrize("a, b, n, exp", td1b)
def test_integrals(a, b, n, exp):
    qm10.helper1.integrals(a, b)
    assert(True)


def test_a_funct():
    qm10.helper1.a_funct(A)
    assert(True)

def test_density_builder():
    qm10.helper1.density_builder(np.array(A),2)
    pass

