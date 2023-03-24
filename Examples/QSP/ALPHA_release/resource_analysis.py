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
###       RESOURCE ANALYSIS  -  Elementary routines for resource analysis    ###
###                                                                          ###
################################################################################

import numpy    as  np
import cirq




## get_t_counts(circ)
## ---------------------------------------x
##
##  Count T, T^\dagger gates in circuit
##  <circ>. Assumes that <circ> has been 
##  decomposed into Clifford+T.
##

def get_t_counts(circ):

    ## Note that T^\dagger is given as 
    ## a ZPowGate 
    ## (e.g., as the fourth root of a Z)

    n_t     =  0       # number of T
    n_t_inv =  0       # number of T^\dagger

    for moment in circ:
        for gate_op in moment:
            if (gate_op.gate == cirq.T):
                n_t     += 1
            elif (gate_op.gate == cirq.ZPowGate(exponent=-0.25)):
                n_t_inv += 1

    return(n_t,n_t_inv)




## get_gate_depth(circ)
## ---------------------------------------
##
##  Return the gate depth of the circuit
##  <circ>. 
##

def get_gate_depth(circ):

    depth = len(cirq.align_left(circ))

    return(depth)