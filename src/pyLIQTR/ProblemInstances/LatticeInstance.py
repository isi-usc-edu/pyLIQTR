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
from pyLIQTR.ProblemInstances.ProblemInstance  import   ProblemInstance




class LatticeInstance(ProblemInstance):

    def __init__( self, 
                  shape         =  None, 
                  cell          =  None, 
                  cell_subtype  =  'default', 
                  pbcs          =  (False,False),
                  rescale       =  True,
                  td            =  None,
                  td_onsite     =  None,
                  model_prefix  =  "",
                  **kwargs ):

        self._shape         =  shape
        self._rescale       =  rescale

        self._pbcs          =  pbcs

        self._cell          =  cell
        self._cell_subtype  =  cell_subtype

        self._td            =  td
        self._td_onsite     =  td_onsite

        self._model_prefix  =  model_prefix

        if (self._td_onsite is not None):
            self._UC = self._cell( self._td,
                                   onsite=self._td_onsite,
                                   type=self._cell_subtype )
        else:
            self._UC = self._cell(self._td,type=self._cell_subtype)

        self._LT = OperatorLattice( size=self._shape, 
                                    unit=self._UC, 
                                    periodic=self._pbcs, 
                                    type='op-qubit')

        self._ops = self._LT.op_strings()

        self._ops.expand_projectors()
        self._ops.cleanup()

        if self._rescale:
            self._ops.rescale_coeff_norm()

        self._model = self._model_prefix+" - "+self._UC.type_string+"("+self._UC.subtype_string+")"


        super(ProblemInstance, self).__init__(**kwargs)




    def __str__(self):
        return f"{self._model}\n\r\n\tN:\t{self._shape}"



    def n_qubits(self):
        return (self._ops.N_qb)



    def n_lcu_terms(self):
        return(ops.N)



    def matrix(self):
        return(self._ops.matrix())


    def get_alpha(self):
        if self._rescale:
            return(self._ops.alpha)
        else:
            return(self._ops.get_alpha())



    def print_unit_cell(self):
        print(self._UC)



    def draw_lattice( self,number_qubits=True,layout='kamada_kawai',
                            doubled=False,single_spin=True):
        
        draw_lattice_2d( self._LT,
                         with_labels=number_qubits,
                         color_nodes=True,
                         font_weight='light',
                         doubled=doubled,
                         single_spin=single_spin )



    def yield_PauliLCU_Info(self,return_as='arrays',do_pad=0,pad_value=1.0):
        if (return_as == 'arrays'):
            terms = self._ops.terms(do_pad=do_pad,pad_value=pad_value)
        elif (return_as == 'strings'):
            terms = self._ops.strings(do_pad=do_pad,pad_value=pad_value)


#         if do_pad:
#             for __ in range(do_pad):
#  #               yield ((),'I',0)
#                yield ([0],'I',0)


        for term in terms:
            yield term










# ###
# ### Todo: Redefine this in terms of fermionic operators from the outset
# ###       and use JWT/BK etc. to map to spins.  Need to finish my general
# ###       transform code, though.
# ###

# class FermiHubbard(LatticeInstance):

#     def __init__( self, 
#                   shape=(4,4), 
#                   J=-1.0, 
#                   U=4.0, 
#                   cell=SquareLattice, 
#                   cell_subtype='default', 
#                   terms=None, 
#                   pbcs = (False,False),
#                   rescale = True,
#                   **kwargs ):

#         self._model_prefix = "Fermi-Hubbard Model"

#         self.shape = shape

#         self.J  =  J
#         self.U  =  U

#         if terms is None:

#             self._td       =  {}
#             self._td['A']  =  [ [ 'Z' ], [ -U/4 ] ]
#             self._td['_']  =  [ [ 'XZX','YZY' ], [ -J/2, -J/2 ] ]

#             self._td_onsite       =  {}
#             self._td_onsite['_']  =  [ [ 'ZZ' ], [ U/8 ] ]

#         else:
#             self._td = terms


#         LatticeInstance.__init__(   self, 
#                                     shape         =  shape, 
#                                     cell          =  cell, 
#                                     cell_subtype  =  cell_subtype, 
#                                     td            =  self._td, 
#                                     td_onsite     =  self._td_onsite,
#                                     pbcs          =  pbcs,
#                                     rescale       =  True,
#                                     model_prefix  =  self._model_prefix,
#                                     **kwargs )





# class Heisenberg(LatticeInstance):

#     def __init__( self, 
#                   shape=(4,4), 
#                   J=(1.0,1.0,1.0,), 
#                   h=(0.5,0.0,0.5), 
#                   cell=SquareLattice, 
#                   cell_subtype='default', 
#                   terms=None, 
#                   pbcs = (False,False),
#                   rescale = True,
#                   **kwargs ):

#         self._model_prefix = "Heisenberg Model"

#         if terms is None:

#             site_cons, site_ops = get_nonzero_match(h,['X','Y','Z'])
#             edge_cons, edge_ops = get_nonzero_match(J,['XX','YY','ZZ'])

#             self._td       =  {}
#             self._td['A']  =  [ site_ops, site_cons ]
#             self._td['_']  =  [ edge_ops, edge_cons ]

#         else:
#             self._td = terms


#         LatticeInstance.__init__(   self, 
#                                     shape         =  shape, 
#                                     cell          =  cell, 
#                                     cell_subtype  =  cell_subtype, 
#                                     td            =  self._td, 
#                                     pbcs          =  pbcs,
#                                     rescale       =  True,
#                                     model_prefix  =  self._model_prefix,
#                                     **kwargs )






# class Transverse_Field_Ising(Heisenberg):

#     def __init__( self, 
#                   J=1.0, 
#                   h=0.0, 
#                   g=-1.0,
#                   terms=None, 
#                   **kwargs ):

#         self._model_prefix = "Transverse Field Ising Model"

#         heisenberg_model.__init__(  self, 
#                                     J=(0.0,0.0,J), 
#                                     h=(g,0.0,h), 
#                                     **kwargs )



