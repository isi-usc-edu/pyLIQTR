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

import cirq
import numpy as np
import pyLIQTR.circuits.pyLOperator as pyLO
from typing import List,Optional,Tuple

class UnitaryBlockEncode(pyLO.pyLOperator):
        def __init__(self, selectOracle, prepareOracle,\
                hamiltonian, phase_qubit,target_qubits, control_qubits, ancilla_qubits):
            self.__hamiltonian = hamiltonian
            self.__phs_q = phase_qubit    # this is the global control (None if there is none)
            self.__tgt_q = target_qubits
            self.__ctl_q = control_qubits
            self.__anc_q = ancilla_qubits

            self.__selOracle = selectOracle
            self.__prepOracle = prepareOracle

            self.__strName = "UnitaryBE"
            
            super(UnitaryBlockEncode, self).__init__()

            #our total decomposition depends on our children decomposition level
            #Either way, we add one level
            self.total_decomp = max([self.__selOracle.total_decomp, \
                                     self.__prepOracle.total_decomp ]) +1
        
        def __str__(self) -> str:
            allQ = [*self.__phs_q, *self.__anc_q, *self.__ctl_q, *self.__tgt_q]
            qStr = ",".join([str(x) for x in allQ])
            return "{} {}".format(self.__strName, qStr)

        def __eq__(self, other):
            if self.__class__ != other.__class__:
                return False
            else:
                return ((self.__hamiltonian == other.__hamiltonian) and \
                    (self.__phs_q == other.__phs_q) and \
                    (self.__ctl_q == other.__ctl_q)and \
                    (self.__tgt_q == other.__tgt_q) and \
                    (self.__anc_q == other.__anc_q) and \
                    (self.__selOracle == other.__selOracle) and \
                    (self.__prepOracle == other.__prepOracle)
                )
        def _circuit_diagram_info_(self, args) -> List[str]:
            return [self.__strName] * self.num_qubits()
        
        def _num_qubits_(self):
            return max([self.__selOracle.num_qubits(), self.__prepOracle.num_qubits()])

        def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
            args.validate_version('2.0')
            allQ = [*self.__phs_q, *self.__anc_q, *self.__ctl_q, *self.__tgt_q]
            allQStr = ",".join([args.format(str(x)) for x in allQ])
            return "{}({})\n".format(self.__strName,allQStr)
        
        def _decompose_(self,qubits):
            #so in general, what does
            #what do we do.
            yield self.__prepOracle.on(*self.__ctl_q)
            yield self.__selOracle.on(*(self.__phs_q + self.__tgt_q + self.__ctl_q + self.__anc_q))
        
        def _get_as_circuit(self):
            allQ = [*self.__phs_q, *self.__anc_q, *self.__ctl_q, *self.__tgt_q]
            return cirq.Circuit(self.on(*allQ))

        def count_exact(self,gate_precision=1e-8) -> pyLO.Dict[str,int]:
            gate_dict = {}
            prep = self.__prepOracle.count_exact()
            sel = self.__selOracle.count_exact()

            for key in prep:
                gate_dict[key] = prep[key]
                
            for key in sel:
                if key in gate_dict:
                    gate_dict[key] += sel[key]
                else:
                    gate_dict[key] = sel[key]

            return gate_dict

class SzegedyWalkOperator(pyLO.pyLOperator):
    pass