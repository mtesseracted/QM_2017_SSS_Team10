import numpy as np
import psi4
from qm10 import helper1

def update_D(F, A, nel):
    '''
    This function updates the density matrix(D) for the next iteration

    Arguments |  Datatype      | Description
              |                |
    diag      | function       | Diagonalizes the core Hamiltonian
    F         | numpy array    | Fock matrix
    A         | numpy array    | Transformation matrix
    nel       | integer        | valence electron count

    Returns
    eps       | numpy array    | Eigenvalues of core Hamiltonian
    D         | numpy array    | Density matrix
    C         | numpy array    | Orthonormal eigenvectors of core Hamiltonian
    nel       |    int         | number of paired electrons
    '''
    eps, C = helper1.diag(F, A)
    Cocc = C[:, :nel]
    D = Cocc @ Cocc.T
    return D, eps


def damping_func(iteration, damp_start, F_old, F_new, damp_value):
    '''
    This function updates the fock matrix by a damping function

    Arguments |  Datatype      | Description 
              |                |
    iteration |     int        | Index that label the loop
    damp_start|     int        | Number of iterations to start the damp procedure
    damp_value|    float       | weight used in the damp procedure
    F_old     | numpy array    | Fock matrix from previous iteration
    F_new     | numpy array    | Fock matrix from current  iteration

    Returns
    F         | nympy array    | Fock matrix after damp function
    '''
    if iteration >= damp_start:
        F = damp_value * F_old + (1.0 - damp_value) * F_new
    else:
        F = F_new
    return F


def gradient(F, D, S):
   '''
   This functions returns the gradient and its root mean square 


    Arguments |  Datatype      | Description 
              |                |
    F         | numpy array    | Fock matrix
    D         | numpy array    | Density matrix
    S         | numpy array    | Overlap matrix

    Returns
    grad      | numpy array    | Gradient
    grad_rms  | numpy.float64  | root mean square of the gradient
   '''
   grad = F @ D @ S - S @ D @ F
   grad_rms = np.mean(grad ** 2) ** 0.5
   return grad, grad_rms


def energy_conv(F, H, D, E_old, mol):
   '''
   This function builds the electronic energy from the Fock, density 
   and core hamiltonian matrices

   Arguments                 |  Datatype      | Description 
                             |                |
      F                      | numpy array    | Fock matrix
      H                      | numpy array    | 
      D                      | numpy array    | Density matrix
      E_old                  | numpy.float64  | Energy from the previous iteration
      E_diff                 | numpy.float64  | Energy difference between current and previous iteration   
 mol.nuclear_repulsion_energy| float          | nuclear_repulsion energy(it is extracted from 
                             |                | the mol <molecule> class)
     mol                     | class          | It is the class that contains the all the information about the molecule 
   '''
   E_electric = np.sum((F + H) * D)
   E_total = E_electric + mol.nuclear_repulsion_energy()
   E_diff = E_total - E_old
   E_old = E_total
   return E_total, E_diff


