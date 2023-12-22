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
from pyLIQTR.circuits.pyLOperator import pyLOperator

from pyLIQTR.circuits.operators.multiCZ import MultiCZ
from cirq import LineQubit, Rx, inverse
import cirq
from typing import List, Optional, Tuple, Dict
from pyLIQTR.utils.circuit_decomposition import decompose_once
from pyLIQTR.gate_decomp.cirq_transforms import clifford_plus_t_direct_transform, get_approximate_t_depth

class Reflect(pyLOperator):
    """
    Implements the Reflect operator.
    """
    def __init__(self, phi:float, phase_qubit:LineQubit, control_qubits:List[LineQubit], ancilla_qubits:List[LineQubit]):
        """Initializes Reflect operator.

        Args:
            phi (float): Angle theta in degrees.
            phase_qubit (LineQubit): A single phase qubit, not a list.
            control_qubits (List[LineQubit]): A list of control qubits.
            ancilla_qubits (List[LineQubit]): A list of ancilla qubits.
        """
        super(Reflect, self).__init__()
        self.__angle = phi #IN DEGREES
        self.__phs_q = phase_qubit
        self.__ctl_q = control_qubits
        self.__anc_q = ancilla_qubits
        
        self.allQ = [*self.__anc_q, *self.__ctl_q, *[self.__phs_q]]
        self.total_decomp = 3
        

    def __str__(self) -> str:
        qStr = ",".join([str(x) for x in self.allQ])
        return f"Reflect ({qStr})"
    
    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        args.validate_version('2.0')
        return str(self)
    
    def _num_qubits_(self):
        return (len(self.__ctl_q) + len(self.__anc_q) + 1) 
    
    def _circuit_diagram_info_(self, args):
        return ["Reflect".format(self.__angle)] * self.num_qubits()
    
    def _decompose_(self, qubits):
        rotation = self.__angle
        
        yield MultiCZ(control_qubits=self.__ctl_q, 
                      target_qubit=[self.__phs_q],
                      ancilla_qubits=self.__anc_q).\
                      on(*([self.__phs_q] + self.__ctl_q + self.__anc_q))

        yield Rx(rads = rotation).on(self.__phs_q)

        yield inverse(MultiCZ(control_qubits=self.__ctl_q, 
                      target_qubit=[self.__phs_q],
                      ancilla_qubits=self.__anc_q).\
                      on(*([self.__phs_q] + self.__ctl_q + self.__anc_q)))
        
    def __eq__(self, other):
        """
        Checks to see if another pyLOperator is equal to this instantiation.
        """
        if self.__class__ != other.__class__:
            return False
        else:
            return (
                (self.__phs_q == other.__phs_q) and \
                (self.__ctl_q == other.__ctl_q) and \
                (self.__anc_q == other.__anc_q)
            )
            
    def _get_as_circuit(self):
        return cirq.Circuit(self.on(*self.allQ))
    