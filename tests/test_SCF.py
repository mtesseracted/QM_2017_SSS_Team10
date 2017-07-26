"""
Test for Hartree Fock SCF code
"""
import qm10
import pytest


# This is a bs function
def test_bsf():
    assert(1 == 1)

def test_add():
    assert qm10.fock.add(5, 2) == 7
    assert qm10.fock.mult(2, 5) == 10

testdata1 = [
    (2, 5, 10),
    (1, 2, 2)
    ]

testdata2 = [
    (4, 2),
    (0, 0),
    (-5, 0)
    ]
@pytest.mark.parametrize("a, b, expected", testdata1)
def test_mult(a, b, expected):
    assert qm10.fock.mult(a, b) == expected
