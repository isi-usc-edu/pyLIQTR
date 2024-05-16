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
import  numpy as np

import  cirq
import  qualtran                     as  qt
import  qualtran.cirq_interop.testing  as  qt_test

import  qualtran.bloqs.hubbard_model as  qt_hm


from    pyLIQTR.BlockEncodings                 import   VALID_ENCODINGS
from    pyLIQTR.BlockEncodings.BlockEncoding   import   BlockEncoding_select_prepare






class fermi_hubbard_square_encoding(BlockEncoding_select_prepare):

    def __init__(self,ProblemInstance, **kwargs):

        super().__init__(ProblemInstance,**kwargs)

        if (self.PI._model != 'Fermi-Hubbard Model - SquareLattice(regular)'):
            raise NotImplementedError

        self.dims = ProblemInstance.shape

        self._encoding_type  =  VALID_ENCODINGS.FermiHubbardSquare

        self._select_gate    =  qt_hm.SelectHubbard( x_dim=self.dims[0], 
                                                       y_dim=self.dims[1], 
                                                       control_val=self._control_val )
   
        self._prepare_gate   =  qt_hm.PrepareHubbard( x_dim=self.dims[0], 
                                                        y_dim=self.dims[1], 
                                                        t=-ProblemInstance.J,
                                                        mu=ProblemInstance.U )
        

    @property
    def alpha(self):
        N  =  2*np.prod(self.PI.shape)
        J  =  np.abs(self.PI.J)
        U  =  np.abs(self.PI.U)
        alpha = 2*N*J + U*N/2
        return (alpha)


