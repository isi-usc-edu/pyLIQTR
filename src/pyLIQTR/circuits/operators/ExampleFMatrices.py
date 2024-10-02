"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""


import cirq
import math
from typing import Sequence, Tuple
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran import Register, QAny, Signature, GateWithRegisters
from functools import cached_property

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
        ancilla = Register('ancilla', QAny(bitsize=self.vector[1]))
        data = Register('data', QAny(bitsize=self.vector[0]))
        control = Register('control', QAny(bitsize=2))
        return Signature([ancilla, data, control])


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
        ancilla = Register('ancilla', QAny(bitsize=self.vector[1]))
        data = Register('data', QAny(bitsize=self.vector[0]))
        control = Register('control', QAny(bitsize=2))
        return Signature([ancilla, data, control])
    

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
        ancilla = Register('ancilla', QAny(bitsize=max(self.vector[1],self.vector[2])))
        data = Register('data', QAny(bitsize=2*self.vector[0]))
        control = Register('control', QAny(bitsize=2))
        return Signature([ancilla, data, control])


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