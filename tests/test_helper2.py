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
A = np.array([[5.0,5.0], [5.0, 5.0]])
F = np.array([[5.0,5.0], [5.0, 5.0]])
F_old = np.array([[5.0,5.0], [5.0, 5.0]])
F_new = np.array([[6.0,6.0], [6.0, 6.0]])
damp_start = 5
iteration = 1
damp_value = 2.0

def test_update_D():
    qm10.helper2.update_D(F, A, 2)
    assert (True)

def test_damping_func():
    qm10.helper2.damping_func(iteration, damp_start, F_old,
    F_new, damp_value)
    assert (True)

D = np.array([[6.0,6.0], [6.0, 6.0]])
S = np.array([[6.0,6.0], [6.0, 6.0]])
def test_gradient():
    qm10.helper2.gradient(F, D, S)

H = np.array([[6.0,6.0], [6.0, 6.0]])
E_old =np.float64(50)

def test_energy_conv():
    qm10.helper2.energy_conv(F, H, D, E_old, mol)
    assert (True)
