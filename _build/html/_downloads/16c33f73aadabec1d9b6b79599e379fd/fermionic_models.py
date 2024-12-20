"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""

###
### Todo: Redefine this in terms of fermionic operators from the outset
###       and use JWT/BK etc. to map to spins.  Need to finish my general
###       transform code, though.

import warnings

import numpy  as  np
import copy   as  copy


from abc   import  ABC, abstractmethod
from enum  import  Enum
from functools import cached_property


from pyLIQTR.clam.lattice_definitions  import  *
from pyLIQTR.clam.operator_lattice     import  OperatorUnitCell, OperatorLattice
from pyLIQTR.clam.operator_lattice     import  draw_lattice_2d
from pyLIQTR.clam.operator_strings     import  op_strings
from pyLIQTR.clam.utilities            import  get_nonzero_match

# class VALID_ENCODINGS(Enum):

#     PauliLCU = 1
#     DPW = 2
#     Fermionic=3

from pyLIQTR.BlockEncodings                    import   VALID_ENCODINGS
from pyLIQTR.ProblemInstances.ProblemInstance  import   ProblemInstance
from pyLIQTR.ProblemInstances.LatticeInstance  import   LatticeInstance

class FermiHubbard(LatticeInstance):

    def __init__( self, 
                  shape=(4,4), 
                  J=-1.0, 
                  U=4.0, 
                  cell=SquareLattice, 
                  cell_subtype='default', 
                  terms=None, 
                  pbcs = (False,False),
                  rescale = True,
                  **kwargs ):

        self._model_prefix = "Fermi-Hubbard Model"

        self.shape = shape

        self.J  =  J
        self.U  =  U

        if terms is None:

            self._td       =  {}
            self._td['A']  =  [ [ 'Z' ], [ U/4 ] ]
            self._td['_']  =  [ [ 'XZX','YZY' ], [ J/2, J/2 ] ]

            self._td_onsite       =  {}
            self._td_onsite['_']  =  [ [ 'ZZ' ], [ U/8 ] ]

        else:
            self._td = terms


        LatticeInstance.__init__(   self, 
                                    shape         =  shape, 
                                    cell          =  cell, 
                                    cell_subtype  =  cell_subtype, 
                                    td            =  self._td, 
                                    td_onsite     =  self._td_onsite,
                                    pbcs          =  pbcs,
                                    rescale       =  True,
                                    model_prefix  =  self._model_prefix,
                                    **kwargs )
        

    def n_qubits(self):
        return(2*np.prod(self.shape))

class FermiHubbardNNN(ProblemInstance):
    """
    This ProblemInstance represents the Fermi-Hubbard model including nearest and next-nearest neighbors terms. The corresponding Hamiltonian can be written as

    .. math::
        -t\\sum_{\\langle p,q\\rangle,\\sigma} a^\\dagger_{p,\\sigma} a_{q,\\sigma}-t'\\sum_{\\langle p,q\\rangle',\\sigma} a^\\dagger_{p,\\sigma} a_{q,\\sigma} + \\frac{u}{2}\\sum_{p,\\alpha\\neq\\beta}n_{p,\\alpha}n_{p,\\beta}

    where the first sum is over nearest neighbors and the second over next-nearest.

    :param Tuple[int] shape: A Tuple containing the dimensions of the lattice. Note, currently only square 2D lattices are supported.
    :param float J1: The nearest neghbor hopping coefficient. Represented as :math:`t` in the above equation.
    :param float J2: The next-nearest neghbor hopping coefficient. Represented as :math:`t'` in the above equation.
    :param float U: Onsite interaction strength. Represented as :math:`u` in the above equation.
    :param Tuple[bool] pbcs: Determines if periodic boundary conditions are on (True) or off (False) for the corresponding dimension. This will affect which terms are included when using the PauliLCU block encoding. Note that the LinearT block encoding assumes periodic boundary conditions.
    """

    def __init__( self, 
                  shape=(4,4), 
                  J1=-1.0, 
                  J2=-1.0,
                  U=4.0, 
                  pbcs = (False,False),
                  **kwargs ):

        self._model_prefix = "Fermi-Hubbard NNN Model"

        if len(shape) != 2:
            raise ValueError(f"Only 2D lattices currently supported for {self._model_prefix}.")
        elif shape[0] != shape[1]:
            raise ValueError(f"Only square lattices currently supported for {self._model_prefix}.")
            
        self.shape = shape
        self._M_vals = np.array(shape)

        self.J1  =  J1
        self.J2  =  J2
        self.U  =  U
        self.pbcs = pbcs

        self._N = np.prod(shape)   

    @cached_property
    def _ops(self):
        """Instance of clam.operator_strings.op_strings that stores the LCU in terms of Pauli operator strings with corresponding coefficients."""

        _ops = op_strings(N_qb=2*self._N)
        spin_offset = self._N
        Nx = self.shape[0]

        for p in range(self._N):

            # ZZ term
            ## alpha=up, beta=down, p = q, factor of 2 for alpha=down, beta=up
            _ops.append_tuple(((p,p+spin_offset), 'ZZ', 2*self.U/8))

            # Z term
            ## omega = up
            _ops.append_tuple(((p,),'Z',-self.U/4))
            ## omega = down
            _ops.append_tuple(((p+spin_offset,),'Z',-self.U/4))

            # XZX + YZY terms
            # loop over q>p 
            for q in range(p+1,self._N):

                xx_str = 'X' + ('Z'*(abs(p-q)-1)) + 'X'
                yy_str = 'Y' + ('Z'*(abs(p-q)-1)) + 'Y'
                idx_set_up = tuple(np.arange(p,p+abs(p-q)+1,1))
                idx_set_down = tuple(np.arange(p+spin_offset,p+spin_offset+abs(p-q)+1,1))
                
                # nearest neighbors
                if abs(p-q)==Nx: # handles internal vertical edges 
                    # spin up
                    _ops.append_tuple((idx_set_up, xx_str, self.J1/2)) 
                    _ops.append_tuple((idx_set_up, yy_str, self.J1/2))
                    # spin down
                    _ops.append_tuple((idx_set_down, xx_str, self.J1/2)) 
                    _ops.append_tuple((idx_set_down, yy_str, self.J1/2))
                elif self.pbcs[1] and abs(p-q)==(Nx-1)*Nx: # handles vertical edges boundaries
                    # spin up
                    _ops.append_tuple((idx_set_up, xx_str, self.J1/2)) 
                    _ops.append_tuple((idx_set_up, yy_str, self.J1/2))
                    # spin down
                    _ops.append_tuple((idx_set_down, xx_str, self.J1/2)) 
                    _ops.append_tuple((idx_set_down, yy_str, self.J1/2))
                elif p%Nx != (Nx-1) and abs(p-q)==1: # handles internal horizontal edges
                    # spin up
                    _ops.append_tuple((idx_set_up, xx_str, self.J1/2)) 
                    _ops.append_tuple((idx_set_up, yy_str, self.J1/2))
                    # spin down
                    _ops.append_tuple((idx_set_down, xx_str, self.J1/2)) 
                    _ops.append_tuple((idx_set_down, yy_str, self.J1/2))
                elif self.pbcs[0] and p%Nx==0 and abs(p-q)==Nx-1: # handles horizontal pbc
                    # spin up
                    _ops.append_tuple((idx_set_up, xx_str, self.J1/2)) 
                    _ops.append_tuple((idx_set_up, yy_str, self.J1/2))
                    # spin down
                    _ops.append_tuple((idx_set_down, xx_str, self.J1/2)) 
                    _ops.append_tuple((idx_set_down, yy_str, self.J1/2))

                # next nearest neighbors
                if abs(p-q)==Nx+1 and p%Nx != (Nx-1): # handles internal right, down / left, up
                    # spin up
                    _ops.append_tuple((idx_set_up, xx_str, self.J2/2)) 
                    _ops.append_tuple((idx_set_up, yy_str, self.J2/2))
                    # spin down
                    _ops.append_tuple((idx_set_down, xx_str, self.J2/2)) 
                    _ops.append_tuple((idx_set_down, yy_str, self.J2/2))
                elif abs(p-q)==Nx-1 and p%Nx != 0: # handles internal left, down / right, up
                    # spin up
                    _ops.append_tuple((idx_set_up, xx_str, self.J2/2)) 
                    _ops.append_tuple((idx_set_up, yy_str, self.J2/2))
                    # spin down
                    _ops.append_tuple((idx_set_down, xx_str, self.J2/2)) 
                    _ops.append_tuple((idx_set_down, yy_str, self.J2/2))
                elif self.pbcs[0] and p%Nx == (Nx-1) and abs(p-q)== 1: # right side boundary down (left side up)
                    # spin up
                    _ops.append_tuple((idx_set_up, xx_str, self.J2/2)) 
                    _ops.append_tuple((idx_set_up, yy_str, self.J2/2))
                    # spin down
                    _ops.append_tuple((idx_set_down, xx_str, self.J2/2)) 
                    _ops.append_tuple((idx_set_down, yy_str, self.J2/2))
                elif self.pbcs[0] and p%Nx == 0 and abs(p-q)== 2*Nx-1: # left side boundary down (right side up)
                    # spin up
                    _ops.append_tuple((idx_set_up, xx_str, self.J2/2)) 
                    _ops.append_tuple((idx_set_up, yy_str, self.J2/2))
                    # spin down
                    _ops.append_tuple((idx_set_down, xx_str, self.J2/2)) 
                    _ops.append_tuple((idx_set_down, yy_str, self.J2/2))
                elif self.pbcs[1] and p<Nx-1 == (Nx-1) and abs(p-q)== (Nx-1)*Nx+1: # top boundary right
                    # spin up
                    _ops.append_tuple((idx_set_up, xx_str, self.J2/2)) 
                    _ops.append_tuple((idx_set_up, yy_str, self.J2/2))
                    # spin down
                    _ops.append_tuple((idx_set_down, xx_str, self.J2/2)) 
                    _ops.append_tuple((idx_set_down, yy_str, self.J2/2))
                elif self.pbcs[1] and 0<p<=Nx-1 == (Nx-1) and abs(p-q)== (Nx-1)*Nx-1: # top boundary left
                    # spin up
                    _ops.append_tuple((idx_set_up, xx_str, self.J2/2)) 
                    _ops.append_tuple((idx_set_up, yy_str, self.J2/2))
                    # spin down
                    _ops.append_tuple((idx_set_down, xx_str, self.J2/2)) 
                    _ops.append_tuple((idx_set_down, yy_str, self.J2/2))
        
        if self.pbcs[0] or self.pbcs[1]:
            # top right corner diagonal
            p = Nx-1
            q = Nx*(Nx-1)
            xx_str = 'X' + ('Z'*(abs(p-q)-1)) + 'X'
            yy_str = 'Y' + ('Z'*(abs(p-q)-1)) + 'Y'
            idx_set_up = tuple(np.arange(p,p+abs(p-q)+1,1))
            idx_set_down = tuple(np.arange(p+spin_offset,p+spin_offset+abs(p-q)+1,1))
            ## spin up
            # factor of 2 for q<p case
            _ops.append_tuple((idx_set_up, xx_str, self.J2/2)) 
            _ops.append_tuple((idx_set_up, yy_str, self.J2/2))
            ## spin down
            # factor of 2 for q<p case
            _ops.append_tuple((idx_set_down, xx_str, self.J2/2)) 
            _ops.append_tuple((idx_set_down, yy_str, self.J2/2))    

            # top left corner diagonal
            p = 0
            q = Nx**2-1
            xx_str = 'X' + ('Z'*(abs(p-q)-1)) + 'X'
            yy_str = 'Y' + ('Z'*(abs(p-q)-1)) + 'Y'
            idx_set_up = tuple(np.arange(p,p+abs(p-q)+1,1))
            idx_set_down = tuple(np.arange(p+spin_offset,p+spin_offset+abs(p-q)+1,1))
            ## spin up
            _ops.append_tuple((idx_set_up, xx_str, self.J2/2)) 
            _ops.append_tuple((idx_set_up, yy_str, self.J2/2))
            ## spin down
            _ops.append_tuple((idx_set_down, xx_str, self.J2/2)) 
            _ops.append_tuple((idx_set_down, yy_str, self.J2/2))   

        return _ops 

    def __str__(self):
        return f"{self._model_prefix}\r\n\tN:\t{self.shape}\n\tJ1,J2,U:{self.J1,self.J2,self.U}"    

    def n_qubits(self):
        return(2*self._N)

    def get_alpha(self,encoding:str=None):
        if encoding == None or encoding == 'PauliLCU':
            return self._ops.get_coeff_norm()
        elif encoding == 'LinearT':
            # factor of 2 for spin
            return (2*self._N * (np.abs(self.J1) + np.abs(self.J2)) + np.abs(self.U)/2 * self._N) * 2

    def yield_PauliLCU_Info(self,return_as='arrays',do_pad=0,pad_value=1.0):
        """yields parameters used for the PauliLCU block encoding."""

        if (return_as == 'arrays'):
            terms = self._ops.terms(do_pad=do_pad,pad_value=pad_value)
        elif (return_as == 'strings'):
            terms = self._ops.strings(do_pad=do_pad,pad_value=pad_value)

        for term in terms:
            yield term

    def yield_LinearT_Info(self,termSelect:str):
        """yields parameters used for the LinearT block encoding."""
        if not self.pbcs[0] or not self.pbcs[1]:
            warnings.warn('LinearT encoding assumes periodic boundary conditions.', stacklevel=2)
        
        if termSelect == 'T':
            # this corresponds to tilde(T)**2 in https://arxiv.org/pdf/1805.03662.pdf
            J1_sign = 1 if np.signbit(self.J1) else 0
            J2_sign = 1 if np.signbit(self.J2) else 0
            for d in range(self._N):
                if d==1 or d==self.shape[0]:
                    yield (J1_sign,np.abs(self.J1))# factor of 2 since coefficients occur for each xzx/yzy term only once depending on p<q or p>q
                elif d==self.shape[0]+1 or d==self.shape[0]-1:
                    yield (J2_sign,np.abs(self.J2))
                else:
                    yield (0,0)

        if termSelect == 'U':
            term_sign = 0 if np.signbit(self.U) else 1
            for d in range(self._N):
                yield (term_sign,np.abs(self.U/4))

        if termSelect == 'V':
            U_sign = 1 if np.signbit(self.U) else 0
            for d in range(self._N):
                yield (U_sign,np.abs(self.U/8))