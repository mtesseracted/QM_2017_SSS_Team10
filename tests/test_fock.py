"""
Test for Hartree Fock SCF code
"""
import qm10
import pytest


testdata1 = [
    (2, 5, 10),
    (1, 2, 2)
    ]
@pytest.mark.parametrize("a, b, expected", testdata1)
def test_two_elec_energy(a, b, expected):
    assert qm10.fock.two_elec_energy(a) == 1
