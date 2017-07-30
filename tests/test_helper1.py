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
nA = np.array(A)
id2 = np.array([[1.,0.],[0.,1.]])
e2 = np.array([1., 1.])

td1 = [(b1, g1, 2, expected), ]
td1b = [(b1, m1, 2, expected), ]
#td2 = [(nA, nA, id2)]

v1 = np.array([1., 0.])
v2 = np.array([0., 1.])

td2 = [(id2, id2, id2, e2)]


@pytest.mark.parametrize("h1, a1, exp1, exp2", td2)
def test_diag(h1, a1, exp1, exp2):
    ev1, mat1 = qm10.helper1.diag(h1, a1)
    assert(np.array_equal(mat1, exp1))
    assert(np.array_equal(ev1, exp2))
    

"""
@pytest.mark.parametrize("a, b, n, exp", td1b)
def test_integrals(a, b, n, exp):
    qm10.helper1.integrals(a, b)
    pass
"""


@pytest.mark.parametrize("h1, a1, exp1, exp2", td2)
def test_updateD(h1, a1, exp1, exp2):
    a1s = a1.shape[0]
    mat1 = qm10.helper1.updateD(h1,a1,a1s)
    assert(np.array_equal(mat1, exp1))
    mat1 = qm10.helper1.updateD(h1,a1,a1s-1)
    exp1[a1s-1,a1s-1]=0.
    assert(np.array_equal(mat1, exp1))
 
