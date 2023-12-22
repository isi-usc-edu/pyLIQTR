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
import cirq
import numpy as np

def convert_sparse_term(term,sz:int):
    assert(len(term)==3)
    if term == ((), 'I', 0):
        #0 pad.
        return cirq.DensePauliString('I'*sz)
    else:
        #intelligent pad
        pstr = ['I' for __ in range(sz)]
        assert(len(term[0])==len(term[1]))
        for q,p in zip(term[0],term[1]):
            if p not in ['I','X','Y','Z']:
                raise ValueError(f'Unsupported pauli type "{p}"')
            pstr[q]=p
        return cirq.DensePauliString(''.join(pstr),coefficient=term[2])
    

    
def convert_to_dense_pauli_string(term):
    return cirq.DensePauliString(term[0],coefficient=term[1])




def convert_projector(pstr):
    ################################################################################
    ###
    ###   FUNCTION     convert_projector(pstr)
    ###
    ################################################################################
    ###
    ###   DESCRIPTION
    ###
    ###      Converts the Z-eigenstate projectors ('+','-') in the string <pstr>
    ###      into their constituent Pauli operations.  Performs rescaling of
    ###      coefficients to match.
    ###
    ###
    ###
    ###   ARGUMENTS
    ###
    ###      pstr   =   Target Pauli string
    ###
    ###   RETURNS
    ###
    ###      new     =  Modified Pauli string
    ###
    ###   REQUIRES
    ###
    ###      numpy as np
    ###
    proj_set = ('+','-')

    new      =  []
    indices  =  pstr[0];      opstr  =  pstr[1];       coeff  =  pstr[2]

    if ( ('+' not in opstr) and ('-' not in opstr)):
        new.append(pstr)
    else:
        for op in proj_set:
            if (op in opstr):

                idx = opstr.index(op)

                ns_1 = opstr[0:idx]+'I'+opstr[(idx+1):]
                ns_2 = opstr[0:idx]+'Z'+opstr[(idx+1):]
                
                sgn = (-1)**proj_set.index(op)

                nsp_1 = ((indices,ns_1,(1/2)*coeff))
                nsp_2 = ((indices,ns_2,sgn*(1/2)*coeff))

                [new.append(string) for string in convert_projector(nsp_1)]
                [new.append(string) for string in convert_projector(nsp_2)]

                break

    return(new)
