"""
Test cases for helper1.py functions
"""
import qm10
import pytest
import psi4
import numpy as np

psi4.core.be_quiet()


### geom
mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
""")

# Build a molecule
mol.update_geometry()

b1 = psi4.core.BasisSet.build(mol, target="aug-cc-pVDZ")
mints = psi4.core.MintsHelper(b1)

expected = 1
A = mints.ao_overlap()

td1 = [(mol,b1,expected), ]

@pytest.mark.parametrize("a, b, exp", td1)
def test_hartree_fock(a, b, exp):
    qm10.helper1.hartree_fock(b, a)
    assert(True)



@pytest.mark.parametrize("a, b, exp", td1)
def test_integrals(b, a, exp):
    qm10.helper1.integrals(b, a)
    assert(True)


def test_a_funct():
    qm10.helper1.a_funct(A)
    assert(True)

