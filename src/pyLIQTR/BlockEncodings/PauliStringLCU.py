"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import  numpy as np

import  cirq
import warnings
import  qualtran               as  qt

# from  functools import cached_property 

from pyLIQTR.BlockEncodings                 import   VALID_ENCODINGS
#  from pyLIQTR.BlockEncodings.BlockEncoding   import   BlockEncoding
from pyLIQTR.BlockEncodings.BlockEncoding   import   BlockEncoding_select_prepare


from pyLIQTR.circuits.operators.prepare     import   Prepare
from pyLIQTR.utils.pauli_string_manip       import   convert_sparse_term, convert_to_dense_pauli_string
from pyLIQTR.utils.resource_analysis         import   legacy_resource_profile


from pyLIQTR.circuits.operators.select_prepare_pauli  import  prepare_pauli_lcu
from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU
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
                    warnings.warn("Alias sampling preparation with negative coefficients is not supported yet. Circuits and estimates will assume positive coefficients.",stacklevel=2)

            self._prepare_gate = StatePreparationAliasSampling.from_probabilities(
                unnormalized_probabilities=[np.abs(np.real(t.coefficient)) for t in self.getTerms]
                , precision=probability_eps)


    @property
    def getTerms(self):        
        terms    =  [convert_to_dense_pauli_string(term)
                        for term in self.PI.yield_PauliLCU_Info(do_pad=self.n_pad, return_as='strings')]    
        return terms

    @property
    def alpha(self):
        return(self.PI.alpha)
