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

b1s = "sto-3g"

b1 = psi4.core.BasisSet.build(m1, target=b1s)
mints = psi4.core.MintsHelper(b1)

expected = 1
A = mints.ao_overlap()
nA = np.array(A)
id2 = np.array([[1.,0.],[0.,1.]])
e2 = np.array([1., 1.])

td1 = [(b1s, g1, 2, expected), ]
td1b = [(b1, m1, 2, expected), ]
#td2 = [(nA, nA, id2)]

v1 = np.array([1., 0.])
v2 = np.array([0., 1.])

td2 = [(id2, id2, id2, e2)]


@pytest.mark.parametrize("a, b, n, exp", td1)
def test_buildInts_H2O(a, b, n, exp):

    #Call function and fill variables
    nbf, V, T, S, g, A, nuc_e, psi4_e = qm10.helper1.buildInts(a, b)

    fname = "tests/h2o-sto3g.test"
    mode = "load" #change to save if making new test file

    if(mode == "save"): #Save the current config to fnam
        shapes = np.concatenate([V.shape, T.shape, S.shape, 
                 g.shape, A.shape])
        savear = np.concatenate( (scals, shapes, V.ravel(), 
                 T.ravel(), S.ravel(), g.ravel(), A.ravel()) )
        np.savetxt(fname, savear)

    elif(mode == "load"): #load config from fnam and compare
        scals = np.array([nbf, nuc_e, psi4_e])
        loadar = np.loadtxt(fname); #Fill array
        scnt = 3 #how many scalar vals first
        ld_scals = np.array(loadar[:scnt]) #load scalars

        #load the shapes of the arrays in order they were filled
        ld_Vsh = np.array(loadar[scnt:scnt+2], dtype=np.int)
        ld_Tsh = np.array(loadar[scnt+2:scnt+4], dtype=np.int)
        ld_Ssh = np.array(loadar[scnt+4:scnt+6], dtype=np.int)
        #g is 4 index
        ld_gsh = np.array(loadar[scnt+6:scnt+10], dtype=np.int)         
        ld_Ash = np.array(loadar[scnt+10:scnt+12], dtype=np.int)


        #Get index positions in loadar from shapes
        vstart = scnt+12
        vend = ld_Vsh[0]*ld_Vsh[1] + vstart
        ld_V = loadar[vstart:vend].reshape(ld_Vsh[0], ld_Vsh[1])

        tend = ld_Tsh[0]*ld_Tsh[1] + vend
        ld_T = loadar[vend:tend].reshape(ld_Tsh[0], ld_Tsh[1])

        send = ld_Ssh[0]*ld_Ssh[1] + tend
        ld_S = loadar[tend:send].reshape(ld_Ssh[0], ld_Ssh[1])

        gend = ld_gsh[0]*ld_gsh[1]*ld_gsh[2]*ld_gsh[3] + send
        ld_g = loadar[send:gend].reshape(ld_gsh[0], ld_gsh[1],
                ld_gsh[2], ld_gsh[3])

        aend = ld_Ash[0]*ld_Ash[1] + gend
        ld_A = loadar[gend:aend].reshape(ld_Ash[0], ld_Ash[1])

        #Check arrays are ==
        assert(np.allclose(ld_scals, scals)) 
        assert(np.allclose(ld_V, V)) 
        assert(np.allclose(ld_T, T))
        assert(np.allclose(ld_S, S))
        assert(np.allclose(ld_g, g))
        assert(np.allclose(ld_A, A))

    pass


@pytest.mark.parametrize("h1, a1, exp1, exp2", td2)
def test_diag(h1, a1, exp1, exp2):
    ev1, mat1 = qm10.helper1.diag(h1, a1)
    assert(np.array_equal(mat1, exp1))
    assert(np.array_equal(ev1, exp2))
    

@pytest.mark.parametrize("h1, a1, exp1, exp2", td2)
def test_updateD(h1, a1, exp1, exp2):
    a1s = a1.shape[0]
    mat1 = qm10.helper1.updateD(h1,a1,a1s)
    assert(np.array_equal(mat1, exp1))
    mat1 = qm10.helper1.updateD(h1,a1,a1s-1)
    exp1[a1s-1,a1s-1]=0.
    assert(np.array_equal(mat1, exp1))


def test_buildF():
    g = np.identity(4,dtype=np.float).reshape(2,2,2,2)
    id2 = np.array([[1.,0.],[0.,1.]])
    o2 = np.ones((2,2))

    Fold, F = qm10.helper1.buildF(g, id2, id2, 0., o2)
    assert(np.allclose(id2, Fold))
    assert(np.allclose(id2, F))

    fac = 0.5
    Fold, F = qm10.helper1.buildF(g, id2, id2, fac, o2)
    assert(np.allclose(fac*(id2+o2), F))
    assert(np.allclose(id2, Fold))

    pass
 

def test_buildE():
    id2 = np.identity(2, dtype=np.float)
    et, ed = qm10.helper1.buildE(id2, id2, id2, 1., 1.)
    assert(np.sum(2.*id2) + 1. == et)
    assert(ed == 4.)

    pass

