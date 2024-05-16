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
import  qualtran               as  qt

# from  functools import cached_property 

from pyLIQTR.BlockEncodings                 import   VALID_ENCODINGS
#  from pyLIQTR.BlockEncodings.BlockEncoding   import   BlockEncoding
from pyLIQTR.BlockEncodings.BlockEncoding   import   BlockEncoding_select_prepare


from pyLIQTR.circuits.operators.prepare     import   Prepare
from pyLIQTR.utils.pauli_string_manip       import   convert_sparse_term, convert_to_dense_pauli_string
from pyLIQTR.utils.resource_analysis         import   legacy_resource_profile


from pyLIQTR.circuits.operators.select_prepare_pauli  import  prepare_pauli_lcu
from qualtran.bloqs.select_pauli_lcu import SelectPauliLCU
from qualtran.bloqs.state_preparation import StatePreparationAliasSampling




class PauliStringLCU(BlockEncoding_select_prepare):


    def __init__(self,ProblemInstance, prepare_type=None, probability_eps=0.002, **kwargs):

        super().__init__(ProblemInstance,**kwargs)

        self._encoding_type     =  VALID_ENCODINGS.PauliLCU

        ##  n_terms   :   Number of problem terms
        ##  n_pad     :   Number of padding terms
        ##  n_tot     :   Total number of terms
        ##
        self.n_terms  =  self.PI.n_terms(encoding=self._encoding_type)
        self.n_pad    =  2**(int(np.ceil(np.log2(self.n_terms)))) - self.n_terms
        self.n_tot    =  self.n_terms + self.n_pad

        alphas        =  [np.sqrt(np.abs(t.coefficient)) for t in self.getTerms]

        selection_bitsize = int(np.ceil(np.log2(self.n_tot)))

        self._select_gate          =  SelectPauliLCU( selection_bitsize=selection_bitsize,
                                                             target_bitsize=self.PI.n_qubits(),
                                                             select_unitaries=self.getTerms,
                                                             control_val=self._control_val )

        if prepare_type is None:
            self._prepare_gate         =  prepare_pauli_lcu(selection_bitsize=selection_bitsize,
                                                         alphas=alphas)
        elif prepare_type=='AS':
            for t in self.getTerms:
                coeff = np.real(t.coefficient)
                if coeff < 0:
                    raise ValueError("Negative coefficients are not supported yet.")
            
            self._prepare_gate         =  StatePreparationAliasSampling.from_lcu_probs(
                                                lcu_probabilities=[np.real(t.coefficient) for t in self.getTerms]
                                                    , probability_epsilon=probability_eps)


    @property
    def getTerms(self):        
        terms    =  [convert_to_dense_pauli_string(term)
                        for term in self.PI.yield_PauliLCU_Info(do_pad=self.n_pad, return_as='strings')]    
        return terms

    @property
    def alpha(self):
        return(self.PI.alpha)
