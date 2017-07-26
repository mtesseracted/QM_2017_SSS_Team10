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
F = 5.0

def test_fork_diag():
    qm10.helper2.fock_diag(A, F)
    assert (True)

F_old = 5.0
F_new = 6.0
iteration = 1.0
damp_value = 2.0
def test_damping_func():
    qm10.helper2.damping_func(iteration, damp_value, F_old, F_new)
    assert (True)

D = 10.0
S = 1.0
def test_gradient_calculation():
    qm10.helper2.gradient_calculation(F, D, S)
    assert (True)

H = 0.0
E_old =2
def test_energy_conv():
    qm10.helper2.energy_conv(F, H, D, E_old)
    assert (True)
