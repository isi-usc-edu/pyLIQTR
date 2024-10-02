"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
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
