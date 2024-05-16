"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
findings, conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the Under Secretary of Defense
for Research and Engineering.

© 2022 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
above. Use of this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.
"""

import cirq
import math
from typing import Sequence, Tuple
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran import Signature, GateWithRegisters
from cirq._compat import cached_property

class F0_operator(GateWithRegisters):
    """
    """
    def __init__(self, n : int = 2, a0_in : int = 2, alpha0 : int = 1, T : TComplexity = TComplexity(t=0,clifford=0)):
        self.n = n
        self.T = T
        self.a0_in = a0_in
        self.alpha0 = alpha0


    @cached_property
    def vector(self) -> Tuple[int]:
        return (self.n, self.a0_in, self.alpha0)

    
    @cached_property 
    def signature(self) -> Signature:
        return Signature.build(ancilla = self.vector[1], data = self.vector[0], control = 2)


    def _t_complexity_(self) -> TComplexity:
        return self.T
    

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.signature:
            if reg.name == "ancilla":
                wire_symbols += ["ancilla"] * reg.total_bits()
            elif reg.name == "data":
                wire_symbols += ["F0"] * reg.total_bits()
            elif reg.name == "control":
                wire_symbols += ["control"] * reg.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)
    

    def __repr__(self) -> str:
        return f"F0"
    

    def _t_complexity_(self) -> TComplexity:
        return TComplexity(t=0, clifford=0)
    

    def decompose_from_registers(
            self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:

        data = quregs["data"]
        ancilla = quregs["ancilla"]
        control = quregs["control"]

        yield cirq.I.on(data[0])
        yield cirq.I.on(ancilla[0])
        yield cirq.I.on(control[0])


class F1_operator(GateWithRegisters):
    """
    """
    
    def __init__(self, n : int = 2, a1 : int = 2, alpha1 : int = 1, T : TComplexity = TComplexity(t=0,clifford=0)):
        self.n = n
        self.T = T
        self.a1 = a1
        self.alpha1 = alpha1


    @cached_property
    def vector(self) -> Tuple[int]:
        return (self.n, self.a1, self.alpha1)

    
    @cached_property 
    def signature(self) -> Signature:
        return Signature.build(ancilla = self.vector[1], data = self.vector[0], control = 2)
    

    def _t_complexity_(self) -> TComplexity:
        return self.T


    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.signature:
            if reg.name == "ancilla":
                wire_symbols += ["ancilla"] * reg.total_bits()
            elif reg.name == "data":
                wire_symbols += ["F1"] * reg.total_bits()
            elif reg.name == "control":
                wire_symbols += ["control"] * reg.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)
    

    def __repr__(self) -> str:
        return f"F1"
    

    def _t_complexity_(self) -> TComplexity:
        return TComplexity(t=0, clifford=0)
    

    def decompose_from_registers(
            self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:

        data = quregs["data"]
        ancilla = quregs["ancilla"]
        control = quregs["control"]

        yield cirq.I.on(data[0])
        yield cirq.I.on(ancilla[0])
        yield cirq.I.on(control[0])


class F2_operator(GateWithRegisters):
    """
    """
    def __init__(self, n : int = 2, a2_in : int = 1, a2_out : int = 2, alpha2 : int = 1, T : TComplexity = TComplexity(t=0,clifford=0)):
        self.n = n
        self.T = T
        self.a2_in = a2_in
        self.a2_out = a2_out
        self.alpha2 = alpha2


    @cached_property
    def vector(self) -> Tuple[int]:
        return (self.n, self.a2_in, self.a2_out, self.alpha2)

    
    @cached_property 
    def signature(self) -> Signature:
        return Signature.build(ancilla = max(self.vector[1], self.vector[2]), data = 2*self.vector[0], control = 2)


    def _t_complexity_(self) -> TComplexity:
        return self.T
    

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.signature:
            if reg.name == "ancilla":
                wire_symbols += ["ancilla"] * reg.total_bits()
            elif reg.name == "data":
                wire_symbols += ["F2"] * reg.total_bits()
            elif reg.name == "control":
                wire_symbols += ["control"] * reg.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)
    

    def __repr__(self) -> str:
        return f"F2"
    

    def _t_complexity_(self) -> TComplexity:
        return TComplexity(t=0, clifford=0)
    

    def decompose_from_registers(
            self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:

        data = quregs["data"]
        ancilla = quregs["ancilla"]
        control = quregs["control"]

        yield cirq.I.on(data[0])
        yield cirq.I.on(ancilla[0])
        yield cirq.I.on(control[0])