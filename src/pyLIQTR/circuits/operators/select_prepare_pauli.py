"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""

import   qualtran   as       qt
#from    pyLIQTR.circuits.operators.prepare     import   Prepare
from qualtran import BoundedQUInt


from    pyLIQTR.circuits.operators.prepare_oracle_pauli_lcu     import  QSP_Prepare  as   Prepare
  



class prepare_pauli_lcu(qt._infra.gate_with_registers.GateWithRegisters):


    def __init__(self, selection_bitsize=3, alphas=None):

        self._selection_bitsize      =   selection_bitsize
        self._alphas                 =   alphas

        self._Prepare                =   Prepare

    @property
    def signature(self):
        sig  = qt._infra.registers.Signature.build_from_dtypes(selection=BoundedQUInt(self._selection_bitsize,len(self._alphas)))
        return(sig)


    @property
    def _prepare_cost(self):
        #Note the analytical expression from https://arxiv.org/pdf/1905.10724.pdf A.3.2 seemed to be 
        #off by a factor of 1/2
        d=self._selection_bitsize
        return qt.cirq_interop.t_complexity_protocol.TComplexity(t=0,\
                                         clifford=2**(d-1),\
                                            rotations=2**(d))


    @property
    def _prepare_gate(self):
        quregs = qt._infra.gate_with_registers.get_named_qubits(self.signature)
        return (self._Prepare(qubit_reg = list(quregs['selection']), alphas=self._alphas))
    

                
    def _t_complexity_(self) -> qt.cirq_interop.t_complexity_protocol.TComplexity:
        prepare_cost  =  self._prepare_cost
        return (prepare_cost)



    def decompose_from_registers(self, context, **quregs):
        yield self._prepare_gate.on(*quregs['selection'])

