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

################################################################################
###                                                                          ###
###    OPERATORS - Matrix-based definition and manipulation of operators     ###
###                                                                          ###
################################################################################



import  numpy                as  np
import  pyLIQTR.clam.multilinear_algebra  as  mla



######                            #############################################
######  Pauli / SU(2) Operators   #############################################
######                            #############################################



## Define Pauli matrices (X/Y/Z) + identity (I)
##

X    =  np.array([ [ 0,   1 ],  [ 1,    0 ] ])
Y    =  np.array([ [ 0, -1j ],  [ 1j,   0 ] ])
Z    =  np.array([ [ 1,   0 ],  [ 0,   -1 ] ])
I    =  np.array([ [ 1,   0 ],  [ 0,    1 ] ])

## Define Hadamard, S-gate, and T-gate
##

H    =  np.array([ [1,    1 ],  [ 1,   -1 ] ])/np.sqrt(2)
S    =  np.array([ [1,    0 ],  [ 0,   1j ] ])/np.sqrt(2)
T    =  np.array([ [1,    0 ],  [ 0,   np.exp(1j*np.pi/4) ] ])

## Pauli ladder operators
##

sig_p  =  (X + 1j*Y)/2
sig_m  =  (X - 1j*Y)/2

## Pauli Z-eigenvalue projectors
##

przp = (I + Z)/2
przm = (I - Z)/2

## Pauli Z eigenstates
##

q_u  =  np.array([1,0])
q_d  =  np.array([0,1])




######                            #############################################
######  Fermionic Operators       #############################################
######                            #############################################




################################################################################
###
###   FUNCTION     fcr(m,N)
###
################################################################################
###
###   DESCRIPTION
###
###      Fermionic creation operator acting on site m in a one dimensional array
###      of N qubits.  The operator is defined using the Jordan-Winger
###      transformation.
###
###   ARGUMENTS
###
###      m     =  site on which creation operator is acting
###      N     =  total number of qubits
###
###   RETURNS
###
###      OP = numpy array corresponding to the creation operator
###
###   REQUIRES
###
###      numpy as np
###

def fcr(m,N):

    opr = 1

    for j in range(m):
        opr = np.kron(opr,Z)

    opr = np.kron(opr,sig_p)

    for j in range(m+1,N):
        opr = np.kron(opr,I)

    return(opr)





################################################################################
###
###   FUNCTION     fan(m,N)
###
################################################################################
###
###   DESCRIPTION
###
###      Fermionic annihilation operator acting on site m in a one dimensional
###      array of N qubits.  The operator is defined using the Jordan-Winger
###      transformation.
###
###   ARGUMENTS
###
###      m     =  site on which annihilation operator is acting
###      N     =  total number of qubits
###
###   RETURNS
###
###      OP = numpy array corresponding to the annihilation operator
###
###   REQUIRES
###
###      numpy as np
###

def fan(m,N):

    opr = (fcr(m,N)).conj().T

    return(opr)





################################################################################
###
###   FUNCTION     state_vacuum(m,N)
###
################################################################################
###
###   DESCRIPTION
###
###      Defines the N-fermion vacuum state via the Joran-Wigner transformation.
###      By convention, we adopt the spin-down product state as the vacuum.
###
###   ARGUMENTS
###
###      N     =  total number of qubits
###
###   RETURNS
###
###      OP = numpy array corresponding to the N-fermion vacuum state
###
###   REQUIRES
###
###      numpy as np
###

def state_vacuum(N):

    #vac_state = mla.mk([q_d for j in range(N)])
    vac_state  = mla.mk([q_d]*N)

    return(vac_state)





################################################################################
###
###   FUNCTION     state_multifermi_idx(occ_idxs,N)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns the state containg m fermions in the sites specified by
###      the indices in occ_idxs.  This is defined by the action of creation
###      operators on the N-fermion vacuum state:
###
###         $c^dag_{i_1} c^dag_{i_2} \dots c^dag_{i_m} |0^N\rangle$
###
###      The states are defined in their spin/qubit analogue via the Jordan-
###      Winger transformation.
###
###   ARGUMENTS
###
###      occ_idxs     =  tuple containing indices of fermions
###      N            =  total number of fermionic sites
###
###   RETURNS
###
###      OP = numpy array corresponding to the state with m fermions in N sites.
###
###   REQUIRES
###
###      numpy as np
###

def state_multifermi_idx(occ_idxs,N):

    wfn = state_vacuum(N)

    for idx in occ_idxs:
        wfn = fcr(idx,N) @ wfn

    return(wfn)


