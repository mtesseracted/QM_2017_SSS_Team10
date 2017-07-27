"""
Test for functions in the helper2.py
"""
import numpy as np
import qm10
import pytest
import psi4

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")
bas = psi4.core.BasisSet.build(mol, target="aug-cc-pVDZ")
mints = psi4.core.MintsHelper(bas)
A = mints.ao_overlap()
nA = np.array(A)
F = 5.0
F_old = 5.0
F_new = 6.0
iteration = 1.0
damp_value = 2.0


def test_damping_func():
    qm10.helper2.damping_func(iteration, damp_value, A, A, 0.2)
    assert (True)

def test_gradient():
    qm10.helper2.gradient(nA, nA, nA)
    assert (True)

H = 0.0
E_old =2
def test_energy_conv():
    qm10.helper2.energy_conv(nA, nA, nA, E_old, mol)
    assert (True)


def test_update_D():
    qm10.helper2.update_D(qm10.helper1.diag, A, np.array(A), 2)
    assert (True)


