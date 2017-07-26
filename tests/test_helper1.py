"""
Test cases for helper1.py functions
"""
import qm10
import pytest
import psi4

psi4.core.be_quiet()


### geom
mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
""")

# Build a molecule
mol.update_geometry()

basis = psi4.core.BasisSet.build(mol, target="aug-cc-pVDZ")

expected = 1
A = 0

td1 = [(mol,basis,expected), ]
@pytest.mark.parametrize("a, b, exp", td1)
def test_hartree_fock(a, b, exp):
    qm10.helper1.hartree_fock(b, a)

    assert(True)

"""
def test_integrals(basis, geom, expected):
    assert(True)


def test_a_funct(A, expected):
    assert(True)
"""

