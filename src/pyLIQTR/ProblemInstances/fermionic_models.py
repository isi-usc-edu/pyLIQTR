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

###
### Todo: Redefine this in terms of fermionic operators from the outset
###       and use JWT/BK etc. to map to spins.  Need to finish my general
###       transform code, though.


import numpy  as  np
import copy   as  copy


from abc   import  ABC, abstractmethod
from enum  import  Enum


from pyLIQTR.clam.lattice_definitions  import  *
from pyLIQTR.clam.operator_lattice     import  OperatorUnitCell, OperatorLattice
from pyLIQTR.clam.operator_lattice     import  draw_lattice_2d
from pyLIQTR.clam.utilities            import  get_nonzero_match

# class VALID_ENCODINGS(Enum):

#     PauliLCU = 1
#     DPW = 2
#     Fermionic=3

from pyLIQTR.BlockEncodings                    import   VALID_ENCODINGS
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