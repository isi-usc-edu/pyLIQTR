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

import   qualtran   as       qt
#from    pyLIQTR.circuits.operators.prepare     import   Prepare
from     pyLIQTR.utils.resource_analysis         import   legacy_resource_profile



from    pyLIQTR.circuits.operators.prepare_oracle_pauli_lcu     import  QSP_Prepare  as   Prepare
  



class prepare_pauli_lcu(qt._infra.gate_with_registers.GateWithRegisters):


    def __init__(self, selection_bitsize=3, alphas=None):

        self._selection_bitsize      =   selection_bitsize
        self._alphas                 =   alphas

        self._Prepare                =   Prepare

    @property
    def signature(self):
        sig  = qt._infra.registers.Signature.build(selection=self._selection_bitsize)
        return(sig)


    @property
    def _prepare_cost(self):
        """
        quregs = cirq_ft.infra.get_named_qubits(self.signature)
        selection_qb   =  list(quregs['selection'])
        return( legacy_resource_profile(self._prepare_gate.on(*selection_qb)) )
        """
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

