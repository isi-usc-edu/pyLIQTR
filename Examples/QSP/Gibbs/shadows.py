"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
findings, conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the Under Secretary of Defense
for Research and Engineering.

© 2023 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
above. Use of this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.
"""

### This file contains functions to help perform operator simulations of classical shadows ###

### General Imports ###
import numpy as np
import pyLIQTR.sim_methods.quantum_ops as qops  
import random

# Useful operators and list of Pauli basis projection operators
had = (2**(-0.5))*np.array([[1,1],[1,-1]])
S = np.array([[1,0],[0,1j]])
id = np.eye(2)
R = S @ had

p_bases = [had, had @ S.conj(), id]


################################################################################
###
###   FUNCTION     expect_set(O, rho)
###
################################################################################
###
###   DESCRIPTION
###
###      Computes expectation values for a set of observables for a given state  
###      
###
###   ARGUMENTS
###
###      O      =  A list which contains the set of observables
###
###      rho    =  state with respect to which we are calculating the observables' 
###                 expectation values
###
###   RETURNS
###
###      expct  =  A list of expectation values for the observables
###                 
###
###
###   REQUIRES
###
###      numpy as np
###

def expect_set(O, rho):
    expct = []
    for o_j in O:
        expct.append(np.trace(np.matmul(o_j, rho)))
    return expct

################################################################################
###
###   FUNCTION     random_unitary(n, e)
###
################################################################################
###
###   DESCRIPTION
###
###      Generates a random unitary matrix. Over many trials, these random 
###      unitaries approximate a unitary 2-design. See Dankert et al.
###
###   ARGUMENTS
###
###      n           =  number of qubits in system
###      e           =  error factor
###
###   RETURNS
###
###      C           =  A unitary matrix
###                 
###
###
###   REQUIRES
###
###      numpy as np
###      qantum_ops as qops
###
### HELPERS:
# C1/P1 Twirl on n qubits as described in Dankert et al
def c1p1(n):
    T = [np.linalg.matrix_power(R, random.randint(0,2)) for i in range(n)]
    return qops.multikron(T)

# Perform a random XOR on the first qubit as described in Dankert et al
def random_XOR(n):
    rxor = np.eye(2**n)
    for i in range(1,n):
        if not int(random.randint(1,4)/4):
            cnot0 = qops.multikron([np.eye(2)]*i + [np.array([[1,0],[0,0]])] + [np.eye(2)]*(n-i-1))
            cnot1 = qops.multikron([qops.px] + [np.eye(2)]*(i-1) + [np.array([[0,0],[0,1]])] + [np.eye(2)]*(n-i-1))
            cnot = np.add(cnot0, cnot1)
            rxor = cnot @ rxor
    return rxor

def random_unitary(n, e):
    C = np.eye(2**n)
    for i in range(int(np.log10(1/e))):
        C = c1p1(n) @ C

        C = random_XOR(n) @ C
        
        C = np.kron(np.eye(2), c1p1(n-1)) @ np.kron(had, np.eye(2**(n-1))) @ C

        C = random_XOR(n) @ C

        C = np.kron(np.eye(2), c1p1(n-1)) @ np.kron(had, np.eye(2**(n-1))) @ C

        if random.randint(0,1):
            C = np.kron(S, np.eye(2**(n-1))) @ C
        
        C = random_XOR(n) @ C
        
        C = np.kron(c1p1(1), np.eye(2**(n-1))) @ C
        
    return C 

################################################################################
###
###   FUNCTION     random_pBasis(n)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns an operator which projects a state into a random Pauli basis,  
###      i.e. each qubit is randomly projected into the X, Y, or Z basis
###
###   ARGUMENTS
###
###      n           =  number of qubits
###
###   RETURNS
###
###      u_list      =  A list of [U_i] which project each qubit into a random Pauli 
###                     basis. Full operator can be calculated by tensoring the list           
###                     together.
###
###   REQUIRES
###
###      random
###

def random_pBasis(n):
    u_list = []
    for i in range(n):
        u_list.append(p_bases[random.randint(0,2)])
    return u_list

################################################################################
###
###   FUNCTION     comp_measurement(rho)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns a pobabilistic measurement outcome for a state
###
###   ARGUMENTS
###
###     rho           =  state represented by a density matrix
###
###   RETURNS
###
###      b            =  A bitstring representing the state post-measurement           
###
###
###   REQUIRES
###
###      random, numpy as np
###

def comp_measurement(rho):
    d = len(rho)
    probs = np.diag(rho)
    choice = random.choices(list(range(d)), weights=probs)[0]
    b = bin(choice)[2:]
    b = '0'*(int(np.log2(d)-len(b))) + b
    return b

################################################################################
###
###   FUNCTION     shadow_estimation(rho, n, O_list)
###
################################################################################
###
###   DESCRIPTION
###
###      Calculates classical shadow estimations for a list of observables wrt a 
###      state rho.
###
###   ARGUMENTS
###
###     rho           =  state represented by a density matrix
###     n             =  number of qubits
###
###   RETURNS
###
###      espct_list      =  A classical shadow state for a given random basis measurement and outcome       
###
###
###   REQUIRES
###
###      random, numpy as np
###

def shadow_estimation(rho, n, O_list):
    # Evolve state with random clifford
    V_list = random.choices([0,1,2], k=n)
    V = qops.multikron([p_bases[i] for i in V_list])
    rhoV = V @ rho @ V.conj().T
    # Measure state in the computational basis
    b = comp_measurement(rhoV)

    expct_list = []
    for O in O_list:
        expct = 1
        for i in range(n):
            if O[i] == 3:
                expct *= 1
            elif O[i] != V_list[i]:
                expct = 0
                break
            else:
                expct *= 3*(-1)**(int(b[i]))
        expct_list.append(expct)
    
    return expct_list

################################################################################
###
###   FUNCTION     med_of_mean(data, K)
###
################################################################################
###
###   DESCRIPTION
###
###      Calculates the median of K means of data subsets 
###
###   ARGUMENTS
###
###     data        =  data for which we want to calculate the median of means
###     K           =  number of subsets of data for which we calculate means
###
###   RETURNS
###
###      median     =  median of the K means of subsets of data       
###
###
###   REQUIRES
###
###      numpy as np
###

def med_of_mean(data, K):
    split_data = np.split(data, K, axis=0)
    means = [np.mean(subset, axis=0) for subset in split_data]    
    return np.median(means, axis=0)
