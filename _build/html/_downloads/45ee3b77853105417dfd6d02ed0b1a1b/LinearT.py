"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import qualtran  as  qt
import cirq

import numpy as np
from functools import cached_property

from pyLIQTR.BlockEncodings import *
from pyLIQTR.BlockEncodings.BlockEncoding import BlockEncoding_select_prepare

from pyLIQTR.circuits.operators.prepare_LinearT import FermionicPrepare_LinearT
from pyLIQTR.circuits.operators.select_LinearT import FermionicSelect_LinearT

class Fermionic_LinearT(BlockEncoding_select_prepare):
    '''
    Implements encoding from section IV of https://arxiv.org/pdf/1805.03662.pdf
    '''
    def __init__(self,ProblemInstance, approx_error=1e-5,energy_error=None,**kwargs):
        super().__init__(ProblemInstance,**kwargs)

        self._encoding_type = VALID_ENCODINGS.LinearT
        self._num_coeffs = 3*2*self.PI._N # factor of 3 for T,U,V. Factor of 2 for spin

        if energy_error is not None:
            self.approx_error = self.get_approx_error(energy_error)
        else:
            self.approx_error = approx_error

        self._select_gate = FermionicSelect_LinearT(self.PI._M_vals,self._control_val)

        self._prepare_gate = FermionicPrepare_LinearT(2*self.getT_array,2*self.getU_array,2*self.getV_array,M_vals=self.PI._M_vals,approx_error=self.approx_error)

    def get_approx_error(self,energy_error,digits=1):
        """calculates upper bound on approx_error given desired precision of estimated energy spectra (energy_error). Based on Eq. 36 in https://arxiv.org/pdf/1805.03662.pdf"""
        mu = np.ceil(np.log2(2*np.sqrt(2)*self.PI.get_alpha('LinearT') / energy_error))
        approx_error_upper_bound = 1/(2**mu * self._num_coeffs)
        power = "{:e}".format(approx_error_upper_bound).split('e')[1]
        return round(approx_error_upper_bound, -(int(power) - digits))

    @cached_property
    def getT_array(self):
        return [coeff for coeff in self.PI.yield_LinearT_Info('T')]

    @cached_property
    def getU_array(self):
        return [coeff for coeff in self.PI.yield_LinearT_Info('U')]

    @cached_property
    def getV_array(self):
        return [coeff for coeff in self.PI.yield_LinearT_Info('V')]

