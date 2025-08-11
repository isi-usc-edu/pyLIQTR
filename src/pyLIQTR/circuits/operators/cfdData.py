"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""

import cirq
import math
import attrs
import numpy as np
from typing import Sequence, Set, Tuple, Optional
from qualtran.drawing import Text, TextBox, WireSymbol
from qualtran.bloqs.mcmt.and_bloq import And
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.bloqs.mcmt.multi_control_multi_target_pauli import MultiControlPauli as mcmtp # remove later
from pyLIQTR.utils.global_ancilla_manager import GlobalQubitManager as manager
from qualtran import Register, QAny, Signature, GateWithRegisters, DecomposeTypeError, CompositeBloq
from qualtran.bloqs.basic_gates import CNOT, XGate, Hadamard, SGate, TGate, Toffoli
from qualtran.cirq_interop import CirqGateAsBloqBase
from pyLIQTR.gate_decomp.rotation_gates import ry_decomp
from functools import cached_property


class Ry(CirqGateAsBloqBase):
    """
    Helper function to wrap ry_decomp in a Bloq class
    """

    def __init__(self, angle : float, eps : float = 1e-10):
        self.angle = angle
        self.eps = eps

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    @cached_property
    def cirq_gate(self) -> cirq.Gate:
        return ry_decomp(rads = self.angle, precision = self.eps)

    def get_gate(self):
        return ry_decomp(rads = self.angle, precision = self.eps)

    def adjoint(self) -> 'Ry':
        return attrs.evolve(self, angle=-self.angle)

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        return TextBox(str(self))


class LCU(GateWithRegisters):
    """
    block-encoding of 
    \[−3(c_i\cdot c_k)(c_i\cdot c_l) + (c_k\cdot cl)\]
    """
    def __init__(self, eps : float = 1e-10):
        self.eps = eps

    @cached_property
    def signature(self) -> Signature:
        data = Register("data", QAny(bitsize = 12))
        ancilla = Register("ancilla", QAny(bitsize = 15))
        return Signature([data, ancilla])
    
    def _t_complexity_(self) -> TComplexity:
        return TComplexity(t = 6 * AMatrix().t_complexity().t
                             + 6 * AMatrix(control = 1).t_complexity().t
                             + vPrep(control = 1).t_complexity().t
                             + CPrep(control = 1).t_complexity().t
                             + 2*(1.15*np.log2(1/self.eps)+9.2)
                          ,clifford = 6 * AMatrix().t_complexity().clifford
                             + 6 * AMatrix(control = 1).t_complexity().clifford
                             + vPrep(control = 1).t_complexity().clifford
                             + CPrep(control = 1).t_complexity().clifford
                             + 4*(1.15*np.log2(1/self.eps)+9.2))
    
    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.signature:
            if reg.name == "data":
                wire_symbols += ["LCU"] * reg.total_bits()
            elif reg.name == "ancilla":
                wire_symbols += ["ancilla"] * reg.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)
    
    def __repr__(self) -> str:
        return f"LCU"
    
    def build_call_graph(self, ssa: "SympySymbolAllocator") -> Set['BloqCountT']:
        return {(Ry(angle=2*np.arccos(1/np.sqrt(13)),eps=self.eps),1),
                (Ry(angle=-2*np.arccos(1/np.sqrt(13)),eps=self.eps),1),
                (AMatrix(),6), (AMatrix(control = 1), 6), (XGate(), 2), (CPrep(control = 1),2),
                (vPrep(control = 1), 1)}
    
    def decompose_from_registers(
            self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        
        data = quregs["data"]
        ancilla = quregs["ancilla"]

        yield Ry(angle=2*np.arccos(1/np.sqrt(13)),eps=self.eps).on(ancilla[0])
        yield cirq.X.on(ancilla[0])
        for idx in range(6):
            yield AMatrix().on(*data[2*idx:2*idx+2],ancilla[idx+1])
        yield vPrep(control=1).on(*data,*ancilla[-2:],ancilla[0])
        yield cirq.X.on(ancilla[0])
        yield CPrep(control=1).on(*data[:6],*ancilla[-2:],ancilla[0])
        yield CPrep(control=1).on(*data[6:],*ancilla[-4:-2],ancilla[0])
        for idx in range(6):
            yield AMatrix(control=1).on(*data[2*idx:2*idx+2],ancilla[idx+6],ancilla[0])
        yield Ry(angle=-2*np.arccos(1/np.sqrt(13)),eps=self.eps).on(ancilla[0])


class vPrep(GateWithRegisters):
    """
    Helper function for zeroing out rows and columns with persistent ancilla and mcx
    """
    def __init__(self, control : int = 1):
        self.control = control

    @cached_property
    def signature(self) -> Signature:
        data = Register("data", QAny(bitsize = 12))
        ancilla = Register("ancilla", QAny(bitsize = 2))
        control = Register("control", QAny(bitsize = 1))
        return Signature([data, ancilla, control])
    
    def build_call_graph(self, ssa: "SympySymbolAllocator") -> Set['BloqCountT']:
        return {(Toffoli(),4), (CNOT(),6), (And(),2), (And(uncompute=True),2), (cH(),12),
                (mcmtp((1,1,1), target_gate=cirq.Z),1), (mcmtp((0,0,0,0,0,0,1), target_gate=cirq.X),1)}

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.signature:
            if reg.name == "data":
                wire_symbols += ["vPrep"] * reg.total_bits()
            elif reg.name == "ancilla":
                wire_symbols += ["ancilla"] * reg.total_bits()
            elif reg.name == "control":
                wire_symbols += ["@"] * reg.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)
    
    def __repr__(self) -> str:
        return f"vPrep"
    
    def decompose_from_registers(
            self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        
        data = quregs["data"]
        ancilla = quregs["ancilla"]
        if self.control == 1: 
            control = quregs["control"][0]
            qm = manager()
            clean = np.array(qm.qalloc(n=6))

            for idx in range(3):
                yield cirq.CCX.on(control, data[2*idx],data[2*idx+6])
                if idx < 2:
                    yield cirq.CX.on(control, data[2*idx])
            yield And().on(control, data[2], clean[0])
            yield cirq.CX.on(clean[0],data[0])
            yield cirq.CX.on(clean[0],data[1])
            yield cirq.CX.on(clean[0],data[4])
            yield And(uncompute=True).on(control, data[2], clean[0])
            yield cirq.CCX.on(control, *data[1:3])
            yield cH().on(data[0],control)
            yield cH().on(data[1],control)
            yield cH().on(ancilla[1],control)
            yield mcmtp((1,1,1), target_gate=cirq.Z).on(control,ancilla[1],*data[:2])
            yield And().on(control,ancilla[1],clean[0])
            yield cH().on(data[1],clean[0])
            yield cH().on(data[0],clean[0])
            yield And(uncompute=True).on(control,ancilla[1],clean[0])
            yield cH().on(ancilla[1],control)
            yield cirq.CX.on(control,ancilla[0])
            yield mcmtp((0,0,0,0,0,0,1), target_gate=cirq.X).on(*data[:6],control,ancilla[0])
            for idx in range(6):
                yield cH().on(data[idx],control)
            qm.qfree(qubits=[*clean])
        else:
            assert(False)


class CPrep(GateWithRegisters):
    """
    Helper function for zeroing out rows and columns with persistent ancilla and mcx
    """
    def __init__(self, control : int = 1):
        self.control = control

    @cached_property
    def signature(self) -> Signature:
        data = Register("data", QAny(bitsize = 6))
        ancilla = Register("ancilla", QAny(bitsize = 2))
        control = Register("control", QAny(bitsize = 1))
        return Signature([data, ancilla, control])
    
    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.signature:
            if reg.name == "data":
                wire_symbols += ["CPrep"] * reg.total_bits()
            elif reg.name == "ancilla":
                wire_symbols += ["ancilla"] * reg.total_bits()
            elif reg.name == "control":
                wire_symbols += ["@"] * reg.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)
    
    def __repr__(self) -> str:
        return f"CPrep"
    
    def decompose_from_registers(
            self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        
        data = quregs["data"]
        ancilla = quregs["ancilla"]
        if self.control == 1: 
            control = quregs["control"][0]
            yield cirq.CX.on(control,ancilla[0])
            yield cirq.CX.on(control,ancilla[1])
            qm = manager()
            clean = np.array(qm.qalloc(n=6))
            for idx in range(4):
                yield cirq.X.on(data[0])
            yield cirq.X.on(data[-1])
            yield And().on(control, data[1],clean[-1])
            yield And().on(clean[-1],data[3],clean[0])
            yield cirq.X.on(data[0])
            yield And().on(data[5],clean[0],clean[1])
            yield And().on(data[0],clean[1],clean[2])
            yield And().on(clean[2],data[2],clean[3])
            yield cirq.X.on(data[2])
            yield And().on(clean[3],data[4],clean[4])
            yield cirq.CX.on(clean[4],ancilla[0])
            yield cirq.CX.on(clean[4],ancilla[1])
            yield And(uncompute=True).on(clean[3],data[4],clean[4])
            yield cirq.X.on(data[4])
            yield cirq.CX.on(clean[2],data[2])
            yield And().on(clean[3],data[4],clean[4])
            yield cirq.CX.on(clean[4],ancilla[0])
            yield cirq.CX.on(clean[4],ancilla[1])
            yield And(uncompute=True).on(clean[3],data[4],clean[4])
            yield And(uncompute=True).on(clean[2],data[2],clean[3])
            yield cirq.X.on(data[2])
            yield cirq.CX.on(clean[1],clean[2])
            yield And().on(clean[2],data[2],clean[3])
            yield And().on(clean[3],data[4],clean[4])
            yield cirq.CX.on(clean[4],ancilla[0])
            yield cirq.CX.on(clean[4],ancilla[1])
            yield And(uncompute=True).on(clean[3],data[4],clean[4])
            yield And(uncompute=True).on(clean[2],data[2],clean[3])
            yield And(uncompute=True).on(data[0],clean[1],clean[2])
            yield And(uncompute=True).on(data[5],clean[0],clean[1])
            yield And(uncompute=True).on(clean[-1],data[3],clean[0])
            yield And(uncompute=True).on(control, data[1],clean[-1])
            for idx in range(5):
                yield cirq.X.on(data[idx+1])
            qm.qfree(qubits=[*clean])  
        else:
            assert(False)


class CMatrix(GateWithRegisters):
    """
    (0.4219,(5,8),0) block-encoding of $C=cc^T$ such that $(C)_{ij}=(c_i\cdot c_j)$.
    Let $x=[1,-1,0]'$ and $o=[1,1,1]'$. Then we can see that choosing
    \begin{align}
        c_x&=o\otimes o\otimes x \\
        c_y&=o\otimes x\otimes o \\
        c_z&=x\otimes o\otimes o
    \end{align}
    corresponds to the 27 unique vectors in a $3\times 3\times 3$ unit cube.
    """
    def __init__(self, control : int = 0):
        self.control = control

    @cached_property
    def signature(self) -> Signature:
        data = Register("data", QAny(bitsize = 6))
        ancilla = Register("ancilla", QAny(bitsize = 8))

        if self.control > 0:
            control = Register("control", QAny(bitsize = self.control))
            return Signature([data, ancilla, control])
        return Signature([data, ancilla])

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.signature:
            if reg.name == "data":
                wire_symbols += ["C"] * reg.total_bits()
            elif reg.name == "ancilla":
                wire_symbols += ["ancilla"] * reg.total_bits()
            elif reg.name == "control":
                wire_symbols += ["@"] * reg.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)
    
    def build_call_graph(self, ssa: "SympySymbolAllocator") -> Set['BloqCountT']:
        if self.control == 2:
            return {(AMatrix(control = 2),6), (Toffoli(),2), (XGate(), 13),
                    (And(), 10), (And(uncompute=True),10), (CNOT(), 8)}
    
    def __repr__(self) -> str:
        return f"C"
    
    def decompose_from_registers(
            self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        
        data = quregs["data"]
        ancilla = quregs["ancilla"]
        if self.control == 2: 
            control = quregs["control"]
            for idx in range(3):
                yield AMatrix(dag = True, control = 2).on(*data[2*idx:(2*idx)+2],ancilla[idx + 1], *control)
            yield cirq.CCX.on(*control,ancilla[0])
            yield cirq.CCX.on(*control,ancilla[4])
            qm = manager()
            clean = np.array(qm.qalloc(n=7))
            for idx in range(4):
                yield cirq.X.on(data[idx])
            yield cirq.X.on(data[-1])
            yield And().on(*control, clean[-1])
            yield And().on(clean[-1],data[1],clean[-2])
            yield And().on(clean[-2],data[3],clean[0])
            yield And().on(data[5],clean[0],clean[1])
            yield And().on(data[0],clean[1],clean[2])
            yield And().on(clean[2],data[2],clean[3])
            yield cirq.X.on(data[2])
            yield And().on(clean[3],data[4],clean[4])
            yield cirq.CX.on(clean[4],ancilla[0])
            yield cirq.CX.on(clean[4],ancilla[4])
            yield And(uncompute=True).on(clean[3],data[4],clean[4])
            yield cirq.X.on(data[4])
            yield cirq.CX.on(clean[2],data[2])
            yield And().on(clean[3],data[4],clean[4])
            yield cirq.CX.on(clean[4],ancilla[0])
            yield cirq.CX.on(clean[4],ancilla[4])
            yield And(uncompute=True).on(clean[3],data[4],clean[4])
            yield And(uncompute=True).on(clean[2],data[2],clean[3])
            yield cirq.X.on(data[2])
            yield cirq.CX.on(clean[1],clean[2])
            yield And().on(clean[2],data[2],clean[3])
            yield And().on(clean[3],data[4],clean[4])
            yield cirq.CX.on(clean[4],ancilla[0])
            yield cirq.CX.on(clean[4],ancilla[4])
            yield And(uncompute=True).on(clean[3],data[4],clean[4])
            yield And(uncompute=True).on(clean[2],data[2],clean[3])
            yield And(uncompute=True).on(data[0],clean[1],clean[2])
            yield And(uncompute=True).on(data[5],clean[0],clean[1])
            yield And(uncompute=True).on(clean[-2],data[3],clean[0])
            yield And(uncompute=True).on(clean[-1],data[1],clean[-2])
            yield And(uncompute=True).on(*control, clean[-1])
            for idx in range(5):
                yield cirq.X.on(data[idx+1])
            for idx in range(3):
                yield AMatrix(control = 2).on(*data[2*idx:(2*idx)+2],ancilla[idx + 5], *control)
            qm.qfree(qubits=[*clean])  
        elif self.control == 0:
            for idx in range(3):
                yield AMatrix(dag = True).on(*data[2*idx:(2*idx)+2],ancilla[idx + 1])
            yield cirq.X.on(ancilla[0])
            yield cirq.X.on(ancilla[4])
            qm = manager()
            clean = np.array(qm.qalloc(n=5))
            for idx in range(4):
                yield cirq.X.on(data[0])
            yield cirq.X.on(data[-1])
            yield And().on(data[1],data[3],clean[0])
            yield cirq.X.on(data[0])
            yield And().on(data[5],clean[0],clean[1])
            yield And().on(data[0],clean[1],clean[2])
            yield And().on(clean[2],data[2],clean[3])
            yield cirq.X.on(data[2])
            yield And().on(clean[3],data[4],clean[4])
            yield cirq.CX.on(clean[4],ancilla[0])
            yield cirq.CX.on(clean[4],ancilla[4])
            yield And(uncompute=True).on(clean[3],data[4],clean[4])
            yield cirq.X.on(data[4])
            yield cirq.CX.on(clean[2],data[2])
            yield And().on(clean[3],data[4],clean[4])
            yield cirq.CX.on(clean[4],ancilla[0])
            yield cirq.CX.on(clean[4],ancilla[4])
            yield And(uncompute=True).on(clean[3],data[4],clean[4])
            yield And(uncompute=True).on(clean[2],data[2],clean[3])
            yield cirq.X.on(data[2])
            yield cirq.CX.on(clean[1],clean[2])
            yield And().on(clean[2],data[2],clean[3])
            yield And().on(clean[3],data[4],clean[4])
            yield cirq.CX.on(clean[4],ancilla[0])
            yield cirq.CX.on(clean[4],ancilla[4])
            yield And(uncompute=True).on(clean[3],data[4],clean[4])
            yield And(uncompute=True).on(clean[2],data[2],clean[3])
            yield And(uncompute=True).on(data[0],clean[1],clean[2])
            yield And(uncompute=True).on(data[5],clean[0],clean[1])
            yield And(uncompute=True).on(data[1],data[3],clean[0])
            for idx in range(5):
                yield cirq.X.on(data[idx+1])
            for idx in range(3):
                yield AMatrix().on(*data[2*idx:(2*idx)+2],ancilla[idx + 5])
            qm.qfree(qubits=[*clean])   
        else:
            assert(False)


class Grover(GateWithRegisters):
    """
    (1,(4,0),0) block-encoding of a standard grover iterate
    \[G_n=\frac{1}{2^{n}}1_n-I\]
    where 1_n is an n-qubit matrix filled with ones.
    """
    def __init__(self, control : int = 0):
        self.control = control

    @cached_property
    def signature(self) -> Signature:
        data = Register("data", QAny(bitsize = 6))

        if self.control > 0:
            control = Register("control", QAny(bitsize = self.control))
            return Signature([data, control])
        return Signature([data])

    def _t_complexity_(self) -> TComplexity:
        if self.control:
            return TComplexity(t=44,clifford=2*6+7*(4+self.control)+5*12+1)
        return TComplexity(t=16,clifford=2*6+7*4+1)
    
    def build_call_graph(self, ssa: "SympySymbolAllocator") -> Set['BloqCountT']:
        if self.control == 0:
            return {(Hadamard(),12),(mcmtp(0,0,0,0,0,target_gate=cirq.Z),1),(XGate(),2)}
        elif self.control == 3:
            return {(And(),2),(And(uncompute=True),2),(cH(),12),(mcmtp((0,0,0,0,0,0,1,1),target_gate=cirq.Z),1)}
        else:
            assert(False)

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.signature:
            if reg.name == "data":
                wire_symbols += ["Grover"] * reg.total_bits()
            elif reg.name == "control":
                wire_symbols += ["@"] * reg.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)
    
    def __repr__(self) -> str:
        return f"Grover"
    
    def decompose_from_registers(
            self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        
        data = quregs["data"]
        if self.control > 0: 
            control = quregs["control"]
            if self.control == 3:
                yield cirq.X.on(control[0])
                yield cirq.X.on(control[1])
                yield cirq.X.on(control[2])
                qm = manager()
                clean = np.array(qm.qalloc(n=2))
                yield And().on(*control[:-1],clean[0])
                yield And().on(clean[0],control[-1],clean[1])
                for idx in range(len(data)):
                    yield cH().on(data[idx],clean[1])
                yield And(uncompute=True).on(clean[0],control[-1],clean[1])
                yield And(uncompute=True).on(*control[:-1],clean[0])
            else:
                assert(False)

            yield cirq.X.on(data[-1])
            yield mcmtp((1,1,1,0,0,0,0,0), target_gate=cirq.Z).on(*control, *data)
            yield cirq.X.on(data[-1])
            
            if self.control == 3:
                yield And().on(*control[:-1],clean[0])
                yield And().on(clean[0],control[-1],clean[1])
                for idx in range(len(data)):
                    yield cH().on(data[idx],clean[1])
                yield And(uncompute=True).on(clean[0],control[-1],clean[1])
                yield And(uncompute=True).on(*control[:-1],clean[0])
                yield cirq.X.on(control[0])
                yield cirq.X.on(control[1])
                yield cirq.X.on(control[2])
                qm.qfree(qubits=[*clean])   

        else:
            for idx in range(len(data)):
                yield cirq.H.on(data[idx])
            
            yield cirq.X.on(data[-1])
            yield mcmtp((0,0,0,0,0), target_gate=cirq.Z).on(*data)
            yield cirq.X.on(data[-1])

            for jdx in range(len(data)):
                yield cirq.H.on(data[jdx])


class WMatrix(GateWithRegisters):
    """
    (1,(0,3),eps) block-encoding of 
    \[{1/4,1/4,1,1}^{\otimes3}\] on the diagonal of a matrix
    """
    def __init__(self, control : bool = False, eps: float = 1e-10):
        self.control = control
        self.eps = eps

    @cached_property
    def signature(self) -> Signature:
        ancilla = Register("ancilla", QAny(bitsize = 3))
        data = Register("data", QAny(bitsize = 6))

        if self.control:
            control = Register("control", QAny(bitsize = 1))
            return Signature([data, ancilla, control])
        return Signature([data, ancilla])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> Set['BloqCountT']:
        if not self.control:
            return {(CNOT(),6),(Ry(angle=np.arccos(1/4),eps=self.eps),3),
                    (Ry(angle=-np.arccos(1/4),eps=self.eps),3)}
        else:
            return {(CNOT(),6),(Ry(angle=np.arccos(1/4),eps=self.eps),3),
                    (Ry(angle=-np.arccos(1/4),eps=self.eps),3), (And(),1),
                    (And(uncompute=True),1)}

    def _t_complexity_(self) -> TComplexity:
        if self.control:
            return TComplexity(t=6.9*np.log2(1/self.eps)+74.1,clifford=13.8*np.log2(1/self.eps)+154.2)
        return TComplexity(t=2.3*np.log2(1/self.eps)+20.7,clifford=4.6*np.log2(1/self.eps)+43.4)
    
    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.signature:
            if reg.name == "ancilla":
                wire_symbols += ["ancilla"] * reg.total_bits()
            elif reg.name == "data":
                wire_symbols += ["W"] * reg.total_bits()
            elif reg.name == "control":
                wire_symbols += ["@"]
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)
    
    def __repr__(self) -> str:
        return f"W"
    
    def decompose_from_registers(
            self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        
        data = quregs["data"]
        ancilla = quregs["ancilla"]
        if self.control: 
            control = quregs["control"][0]
            qm = manager()
            clean = np.array(qm.qalloc(n=1))[0]

            for idx in range(3):
                yield And().on(control,data[2*idx+1],clean)
                yield Ry(angle=np.arccos(1/4), eps = self.eps/2).on(ancilla[idx])
                yield cirq.CX.on(clean,ancilla[idx])
                yield Ry(angle=-np.arccos(1/4), eps = self.eps/2).on(ancilla[idx])
                yield cirq.CX.on(clean,ancilla[idx])
                yield And(uncompute=True).on(control,data[2*idx+1],clean)
            qm.qfree(qubits=[clean])
        else:
            for idx in range(3):
                yield Ry(angle=np.arccos(1/4), eps = self.eps).on(ancilla[idx])
                yield cirq.CX.on(data[2*idx+1],ancilla[idx])
                yield Ry(angle=-np.arccos(1/4), eps = self.eps).on(ancilla[idx])
                yield cirq.CX.on(data[2*idx+1],ancilla[idx])


class AMatrix(GateWithRegisters):
    """
    (0.886,(1,1),0) block-encoding of 
    \[A = 
    \begin{pmatrix}
        1 & 1 & \cdot & \cdot \\
        1 & -1 & \cdot & \cdot \\
        1 & 0 & \cdot & \cdot \\
        \cdot & \cdot & \cdot & \cdot
    \end{pmatrix}\]
    """
    def __init__(self, control : int = 0, dag : bool = False):
        self.control = control
        self.dag = dag

    @cached_property
    def signature(self) -> Signature:
        ancilla = Register("ancilla", QAny(bitsize = 1))
        data = Register("data", QAny(bitsize = 2))

        if self.control > 0:
            control = Register("control", QAny(bitsize = self.control))
            return Signature([data, ancilla, control])
        return Signature([data, ancilla])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> Set['BloqCountT']:
        if self.control == 0:
            return {(Hadamard(), 4), (Toffoli(), 1)}
        elif self.control == 1:
            return {(cH(), 4), (mcmtp((1,1,1),target_gate=cirq.X), 1)}
        elif self.control == 2:
            return {(cH(), 4), (And(), 1), (And(uncompute=True), 1), (mcmtp((1,1,1,1),target_gate=cirq.X), 1)}
        else:
            assert(False)

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.signature:
            if reg.name == "ancilla":
                wire_symbols += ["ancilla"]
            elif reg.name == "data":
                wire_symbols += ["A"] * reg.total_bits()
            elif reg.name == "control": 
                wire_symbols += ["@"] * reg.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)

    def __repr__(self) -> str:
        return f"A"

    def decompose_from_registers(
            self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        
        data = quregs["data"]
        ancilla = quregs["ancilla"][0]
        if self.control == 2: 
            control = quregs["control"]
            qm = manager()
            clean = np.array(qm.qalloc(n=1))[0]
            if self.dag:
                yield And().on(*control,clean)
                yield cH().on(ancilla,clean)
                yield cH().on(data[0],clean)
                yield cH().on(data[1],clean)
                yield mcmtp((1,)*(2+self.control), target_gate=cirq.X).on(ancilla,*control,*data[::-1])
                yield cH().on(clean,ancilla)
                yield And(uncompute=True).on(*control,clean)
            else:
                yield And().on(*control,clean)
                yield cH().on(ancilla,clean)
                yield mcmtp((1,)*(2+self.control), target_gate=cirq.X).on(ancilla,*control,*data[::-1])
                yield cH().on(data[1],clean)
                yield cH().on(data[0],clean)
                yield cH().on(ancilla,clean)
                yield And(uncompute=True).on(*control,clean)
            qm.qfree(qubits=[clean])  
        elif self.control == 1:
            control = quregs["control"][0]
            if self.dag:
                yield cH().on(ancilla,control)
                yield cH().on(data[0],control)
                yield cH().on(data[1],control)
                yield mcmtp((1,)*(2+self.control), target_gate=cirq.X).on(ancilla,control,*data[::-1])
                yield cH().on(clean,control)
            else:
                yield cH().on(ancilla,control)
                yield mcmtp((1,)*(2+self.control), target_gate=cirq.X).on(ancilla,control,*data[::-1])
                yield cH().on(data[1],control)
                yield cH().on(data[0],control)
                yield cH().on(ancilla,control)
        elif self.control == 0:
            if self.dag:
                yield cirq.H.on(ancilla)
                yield cirq.H.on(data[0])
                yield cirq.H.on(data[1])
                yield cirq.CCX.on(ancilla,*data[::-1])
                yield cirq.H.on(ancilla)
            else:
                yield cirq.H.on(ancilla)
                yield cirq.CCX.on(ancilla,*data[::-1])
                yield cirq.H.on(ancilla)
                yield cirq.H.on(data[0])
                yield cirq.H.on(data[1])
        else:
            assert(False)


class cH(GateWithRegisters):
    """
    Decomposition of controlled Hadamard with only two T gates
    """
    @cached_property
    def signature(self) -> Signature:
        control = Register("control", QAny(bitsize = 1))
        data = Register("data", QAny(bitsize = 1))
        return Signature([data, control]) 

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> Set['BloqCountT']:
        return {(SGate(),2),(Hadamard(),2),(CNOT(),1),(TGate(),2)}

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.signature:
            if reg.name == "data":
                wire_symbols += ["H"] * reg.total_bits()
            elif reg.name == "control":
                wire_symbols += ["@"]
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)

    def decompose_from_registers(
            self, * context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:

        data = quregs["data"][0]
        control = quregs["control"][0]

        yield cirq.S.on(data)
        yield cirq.H.on(data)
        yield cirq.T.on(data)
        
        yield cirq.CX.on(control, data)
        
        yield cirq.inverse(cirq.T).on(data)
        yield cirq.H.on(data)
        yield cirq.inverse(cirq.S).on(data)


class F3_operator(GateWithRegisters):
    """
    (3/106496,(6,25),eps) block-encoding of F3 as in arXiv:2406.06323
    """
    def __init__(self, eps : float = 1e-10):
        self.eps = eps

    @cached_property
    def signature(self) -> Signature:
        data = Register("data", QAny(bitsize = 40))
        ancilla = Register("ancilla", QAny(bitsize = 10))
        return Signature([data, ancilla])

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.signature:
            wire_symbols += ["F3"] * reg.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)

    def __repr__(self) -> str:
        return f"F3"

    def decompose_from_registers(
            self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:

        data = quregs["data"]
        ancilla = quregs["ancilla"]

        yield F2_operator(eps = self.eps).on(*data)
        for idx in range(3):
            yield cirq.CCX.on(*ancilla[2*idx:2*idx+2],ancilla[idx+5])
        for jdx in range(6):
            yield cirq.H.on(ancilla[jdx])
        yield cirq.X.on(ancilla[-1])
        yield mcmtp((0,)*6, target_gate=cirq.X).on(*ancilla[:6],ancilla[-1])


class F2_operator(GateWithRegisters):
    """
    (3/13312,(6,22),eps) block-encoding of F2 as in arXiv:2406.06323
    """
    def __init__(self, eps : float = 1e-10):
        self.eps = eps
        self.eps1 = 3*self.eps/5
        self.eps2 = 2*self.eps/5

    @cached_property
    def signature(self) -> Signature:
        data = Register("data", QAny(bitsize = 27))
        ancilla = Register("ancilla", QAny(bitsize = 13))
        return Signature([data, ancilla])

    def _t_complexity_(self) -> TComplexity:
        return TComplexity(t = math.ceil(14*4 + LCU(eps = self.eps1).t_complexity().t
                             + WMatrix(eps = self.eps2).t_complexity().t),
                           clifford = math.ceil(9*14 + 36 + LCU(eps = self.eps1).t_complexity().clifford
                             + WMatrix(eps = self.eps2).t_complexity().clifford))

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.signature:
            wire_symbols += ["F2"] * reg.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)

    def __repr__(self) -> str:
        return f"F2"
    
    def decompose_from_registers(
            self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:

        data = quregs["data"]
        ancilla = quregs["ancilla"]

        for idx in range(6):
            yield cirq.CCX.on(*data[2*idx:2*idx+2],ancilla[idx])
        yield LCU(eps = self.eps1).on(*data)
        yield WMatrix(eps=self.eps2).on(*data[:6],*ancilla[6:9])
        yield cirq.X.on(ancilla[9])
        yield mcmtp((0,)*6,target_gate=cirq.X).on(*data[6:12],ancilla[9])
        for idx in range(3):
            yield cirq.CCX.on(*data[2*idx:2*idx+2],ancilla[-idx-1])


class F1_operator(GateWithRegisters):
    """
    let P be a diagonal projection off of the arbitrary-padded elements such that $(P)_{ii} = 1$
    if $c_i$ corresponds to one of the lattice points in the cube and $(P)_{ii} = 0$ otherwise. 
    This allows us to write
    \[F1 = P(W(16+ 3C)−I6)P\]

    (1/257,(7,20),eps) block-encoding
    """
    def __init__(self, eps : float = 1e-10):
        self.eps = eps
        self.eps0 = 12*self.eps/887.8
        self.eps1 = 2*self.eps/887.8
        self.eps2 = self.eps/887.8

    @cached_property
    def signature(self) -> Signature:
        data = Register("data", QAny(bitsize = 6))
        ancilla = Register("ancilla", QAny(bitsize = 9+11))
        return Signature([data, ancilla])
    
    def build_call_graph(self, ssa: "SympySymbolAllocator") -> Set['BloqCountT']:
        return {(Toffoli(), 6), (XGate(), 6), (Grover(control = 3), 1),
                (Hadamard(), 2), (CMatrix(control = 2), 1), (WMatrix(control = 1), 1),
                (Ry(angle=2*np.arccos(16/np.sqrt(257)),eps=self.eps2), 2),
                (Ry(angle=-2*np.arccos(1/2),eps=self.eps1), 1),
                (Ry(angle=2*np.arccos(1/2),eps=self.eps1), 1)}

    def _t_complexity_(self) -> TComplexity:
        return TComplexity(t = math.ceil(6.9*np.log2(1/self.eps0) + 2.3*np.log2(1/self.eps1)
                             + 2.3*np.log2(1/self.eps2) + 318.9),
                           clifford = math.ceil(13.8*np.log2(1/self.eps0) + 4.6*np.log2(1/self.eps1)
                             + 4.6*np.log2(1/self.eps2) + 6*14 + 2 + WMatrix(control = 1).t_complexity().clifford
                             + Grover(control = 3).t_complexity().clifford + CMatrix(control = 2).t_complexity().clifford)) 
    
    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.signature:
            wire_symbols += ["F1"] * reg.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)
    
    def __repr__(self) -> str:
        return f"F1"
    
    def decompose_from_registers(
            self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:

        data = quregs["data"]
        ancilla = quregs["ancilla"]

        for idx in range(3):
            yield cirq.CCX.on(*data[2*idx:2*idx+2],ancilla[idx+8])

        yield Ry(angle=2*np.arccos(16/np.sqrt(257)),eps=self.eps2).on(ancilla[-3])
        yield Ry(angle=-2*np.arccos(1/2),eps=self.eps1).on(ancilla[-2])
        yield cirq.H.on(ancilla[-1])

        for jdx in range(3):
            yield cirq.X.on(ancilla[-jdx-1])

        yield WMatrix(control=1).on(*data,*ancilla[-6:-2])
        yield Grover(control=3).on(*data,*ancilla[-3:])

        yield cirq.X.on(ancilla[-1])
        
        yield CMatrix(control=2).on(*data,*ancilla[:8],*ancilla[-3:-1])

        yield cirq.X.on(ancilla[-2])
        yield cirq.X.on(ancilla[-3])

        yield cirq.H.on(ancilla[-1])
        yield Ry(angle=2*np.arccos(1/2), eps=self.eps1).on(ancilla[-2])
        yield Ry(angle=2*np.arccos(16/np.sqrt(257)),eps=self.eps2).on(ancilla[-3])

        for idx in range(3):
            yield cirq.CCX.on(*data[2*idx:2*idx+2],ancilla[idx+11])
