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
import math
import numpy as np
from typing import Sequence, Tuple

from cirq._compat import cached_property
from qualtran import Signature, GateWithRegisters, Register, SelectionRegister
from qualtran.bloqs.and_bloq import And 

from pyLIQTR.BlockEncodings.__init__ import VALID_ENCODINGS
from pyLIQTR.BlockEncodings.BlockEncoding import BlockEncoding
from pyLIQTR.utils.global_ancilla_manager import GlobalQubitManager as manager
from pyLIQTR.circuits.operators.swap_network import SwapNetwork
from pyLIQTR.circuits.operators.cyclic_shift import CyclicShiftPermutation
from pyLIQTR.circuits.operators.cascading_translation import CascadingTranslation
from pyLIQTR.circuits.operators.diagonal_bitstring import DiagonalBitstring

class Carleman_Linearization(BlockEncoding):
    """
    """

    def __init__(self, ProblemInstance, K: int, **kwargs):
        
        super().__init__(ProblemInstance, **kwargs)
        self._encoding_type = VALID_ENCODINGS.CarlemanLinearization

        self.n = ProblemInstance.vector[0]
        self.K = ProblemInstance.vector[1]

        if self.K < 2:
            raise ValueError("K = 1 not supported")

        self._F0_gate = ProblemInstance.F0_oracle()
        self._F1_gate = ProblemInstance.F1_oracle()
        self._F2_gate = ProblemInstance.F2_oracle()
    
    
    def control_registers(self) -> Tuple[Register, ...]:
        return (Register("F control", bitsize=2),)


    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        return (SelectionRegister("K selection", bitsize = math.ceil(np.log2(self.K))),)


    def target_registers(self) -> Tuple[Register, ...]:
        return (Register("Tensor Data", bitsize = self.n*self.K+1),)


    @cached_property
    def vector(self) -> Tuple[int]:
        return (self.n, self.K)
    

    @cached_property
    def F_oracles(self) -> Tuple[GateWithRegisters]:
        return (self._F0_gate, self._F1_gate, self._F2_gate)
    

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(ancilla = 7 + max(self.F_oracles[1].vector[1], self.F_oracles[0].vector[1],
                                                 self.F_oracles[2].vector[1], self.F_oracles[2].vector[2])
                               + math.ceil(np.log2(self.vector[1])), data = self.vector[0]*self.vector[1] + 1)


    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.signature:
            wire_symbols += ["Carleman"] * reg.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)


    def __repr__(self) -> str:
        return f"Carleman Linearization"
    

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
        ) -> cirq.OP_TREE:

        data = quregs["data"]
        ancilla = quregs["ancilla"]

        a0_in = self.F_oracles[0].vector[1]
        a1 = self.F_oracles[1].vector[1]
        a2_in = self.F_oracles[2].vector[1]
        a2_out = self.F_oracles[2].vector[2]

        alpha0 = self.F_oracles[0].vector[-1]
        alpha1 = self.F_oracles[1].vector[-1]
        alpha2 = self.F_oracles[2].vector[-1]

        yield LCUPrep(alpha0, alpha1, alpha2, False).on_registers(data=ancilla[:2])
        
        for idx in range(math.ceil(np.log2(self.K))):
            yield cirq.H.on(ancilla[-max(a0_in, a1, a2_in,a2_out)-math.ceil(np.log2(self.K))+idx])

        yield CyclicShiftPermutation(math.ceil(np.log2(self.K)),
                                     "right",(True,2),True).on_registers(data=ancilla[-max(a0_in,a1,a2_in,a2_out)-math.ceil(np.log2(self.K)):-max(a0_in,a1,a2_in,a2_out)],
                                                                         control=ancilla[:2])
        
        yield CascadingTranslation(self.vector[0],self.vector[1],
                                   "right",True).on_registers(data=list(data)+list(ancilla[3:4]),
                                                              ancilla=list(ancilla[4:5])+list(ancilla[-max(a0_in,a1,a2_in,a2_out)-math.ceil(np.log2(self.K)):-max(a0_in,a1,a2_in,a2_out)]))
        
        yield CyclicShiftPermutation(math.ceil(np.log2(self.K)),"left",
                                     (True,2),True).on_registers(data=ancilla[-max(a0_in,a1,a2_in,a2_out)-math.ceil(np.log2(self.K)):-max(a0_in,a1,a2_in,a2_out)],
                                                                 control=ancilla[:2])
        
        yield SwapNetwork(self.n, self.K, 2, True).on_registers(data=data[:-self.n],
                                                                ancilla=ancilla[-max(a0_in, a1, a2_in,a2_out)-math.ceil(np.log2(self.K)):-max(a0_in,a1,a2_in,a2_out)],
                                                                control=ancilla[:2])
        yield SwapNetwork(self.n, self.K, 1, False).on_registers(data=data,
                                                                 ancilla=ancilla[-max(a0_in, a1, a2_in,a2_out)-math.ceil(np.log2(self.K)):-max(a0_in,a1,a2_in,a2_out)],
                                                                 control=ancilla[1])

        yield SwapCorrections(self.n).on_registers(control=ancilla[:2],data=data[-self.n:], ancilla=ancilla[-self.n:])


        if max(a0_in,a1,a2_in,a2_out) == a0_in:
            yield self.F_oracles[0].on_registers(data=data[-self.n:], ancilla = ancilla[-max(a0_in,a1,a2_in,a2_out):], control=ancilla[:2])
        else:
            yield self.F_oracles[0].on_registers(data=data[-self.n:], ancilla = ancilla[-max(a0_in,a1,a2_in,a2_out):-max(a0_in,a1,a2_in,a2_out)+a0_in], control=ancilla[:2])
        if max(a0_in,a1,a2_in,a2_out) == a1:
            yield self.F_oracles[1].on_registers(data=data[-self.n:], ancilla = ancilla[-max(a0_in,a1,a2_in,a2_out):], control=ancilla[:2])
        else:
            yield self.F_oracles[1].on_registers(data=data[-self.n:], ancilla = ancilla[-max(a0_in,a1,a2_in,a2_out):-max(a0_in,a1,a2_in,a2_out)+a1], control=ancilla[:2])
        if max(a0_in,a1,a2_in,a2_out) == max(a2_in,a2_out):
            yield self.F_oracles[2].on_registers(data=data[-2*self.n:], ancilla = ancilla[-max(a0_in,a1,a2_in,a2_out):], control=ancilla[:2])  
        else:
            yield self.F_oracles[2].on_registers(data=data[-2*self.n:], ancilla = ancilla[-max(a0_in,a1,a2_in,a2_out):-max(a0_in,a1,a2_in,a2_out)+max(a2_in,a2_out)], control=ancilla[:2])  


        yield SwapCorrections(self.n).on_registers(control=ancilla[:2], data=data[-2*self.n:-self.n], ancilla=ancilla[-self.n:])
            
        yield LCUCorrections(self.K).on_registers(control=ancilla[:2],data=ancilla[-max(a0_in,a1,a2_in,a2_out)-math.ceil(np.log2(self.K)):-max(a0_in,a1,a2_in,a2_out)],target=ancilla[-1])
            
        yield SwapNetwork(self.n, self.K, 0, False).on_registers(data=data,ancilla=ancilla[-max(a0_in, a1, a2_in,a2_out)-math.ceil(np.log2(self.K)):-max(a0_in,a1,a2_in,a2_out)])
        
        yield CyclicShiftPermutation(math.ceil(np.log2(self.K)),"right",
                                     (True,2),True).on_registers(data=ancilla[-max(a0_in,a1,a2_in,a2_out)-math.ceil(np.log2(self.K)):-max(a0_in,a1,a2_in,a2_out)],
                                                                 control=ancilla[:2])
        
        yield CascadingTranslation(self.vector[0],self.vector[1],
                                   "right",True).on_registers(data=list(data)+list(ancilla[5:6]),
                                                              ancilla=list(ancilla[6:7])+list(ancilla[-max(a0_in,a1,a2_in,a2_out)-math.ceil(np.log2(self.K)):-max(a0_in,a1,a2_in,a2_out)]))
        
        yield CyclicShiftPermutation(math.ceil(np.log2(self.K)),"left",
                                     (True,2),True).on_registers(data=ancilla[-max(a0_in,a1,a2_in,a2_out)-math.ceil(np.log2(self.K)):-max(a0_in,a1,a2_in,a2_out)],control=ancilla[:2])

        for idx in range(math.ceil(np.log2(self.K))):
            yield cirq.H.on(ancilla[-max(a0_in, a1, a2_in,a2_out)-math.ceil(np.log2(self.K))+idx])

        yield LCUPrep(alpha0, alpha1, alpha2, True).on_registers(data=ancilla[:2])

        yield DiagonalBitstring(self.n * self.K + 1, self.vector, True).on_registers(data=data, ancilla=ancilla[2])


class LCUCorrections(GateWithRegisters):
    """
    """

    def __init__(self, K : int):

        if K < 3:
            raise ValueError("not implemented for K < 3")

        self.K = K


    @cached_property
    def signature(self) -> Signature:
        return Signature.build(control = 2, data = math.ceil(np.log2(self.K)),
                               target = 1)
    

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.signature:
            if reg.name == "control":
                wire_symbols += ["control"] * reg.total_bits()
            else:
                wire_symbols += ["mcxCorrections"] * reg.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)
    

    def __repr__(self) -> str:
        return f"MCX Correction Terms"
    

    def decompose_from_registers(
            self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
        ) -> cirq.OP_TREE:

        data = quregs["data"]
        control = quregs["control"]
        target = quregs["target"]
        qm = manager()

        clean = np.array(qm.qalloc(n=math.ceil(np.log2(self.K))+1))
        yield And().on(*control, clean[0])
        yield cirq.CX.on(clean[0],target[0])
        yield And(uncompute=True).on(*control, clean[0])

        yield And().on(*data[:2], clean[-1])
        for idx in range(math.ceil(np.log2(self.K))-2):
            yield And().on(clean[-1-idx], data[2+idx], clean[-2-idx])
        yield And().on(clean[2],control[1],clean[1])
        yield cirq.X.on(control[0])
        yield And().on(clean[1],control[0],clean[0])
        yield cirq.CX.on(clean[0],target[0])
        yield And(uncompute=True).on(clean[1],control[0],clean[0])
        yield cirq.CX.on(clean[2],clean[1])
        yield And(uncompute=True).on(clean[2],control[1],clean[1])
        for jdx in range(math.ceil(np.log2(self.K))-2):
            yield And(uncompute=True).on(clean[3+jdx], data[-1-jdx], clean[2+jdx])
        yield And(uncompute=True).on(*data[:2], clean[-1])
        qm.qfree(qubits=[*clean])


class SwapCorrections(GateWithRegisters):
    """
    """

    def __init__(self, n : int):

        self.n = n


    @cached_property
    def signature(self) -> Signature:
        return Signature.build(control = 2, data = self.n, ancilla = self.n)
    

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.signature:
            if reg.name == "control":
                wire_symbols += ["control"] * reg.total_bits()
            else:
                wire_symbols += ["swapCorrections"] * reg.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)
    

    def __repr__(self) -> str:
        return f"MCX Correction Terms"
    

    def decompose_from_registers(
            self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
        ) -> cirq.OP_TREE:

        data = quregs["data"]
        control = quregs["control"]
        ancilla = quregs["ancilla"]
        qm = manager()

        clean = np.array(qm.qalloc(n=2))

        yield And().on(*control, clean[0])
        for idx in range(self.n):
            yield cirq.CX.on(data[-1-idx], ancilla[-1-idx])
            yield And().on(clean[0], ancilla[-1-idx], clean[1])
            yield cirq.CX.on(clean[1], data[-1-idx])
            yield And(uncompute=True).on(clean[0], ancilla[-1-idx], clean[1])
            yield cirq.CX.on(data[-1-idx], ancilla[-1-idx])
        yield And(uncompute=True).on(*control, clean[0])

        qm.qfree(qubits=[*clean])


class LCUPrep(GateWithRegisters):
    """
    """

    def __init__(self, alpha0 : int, alpha1 : int, alpha2 : int, adjoint : bool):

        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        
        self.adjoint = adjoint


    @cached_property
    def vector(self) -> Tuple[int]:
        return (self.alpha0, self.alpha1, self.alpha2)
    

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(data = 2)
    

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.signature:
            if not self.adjoint:
                wire_symbols += ["prep"] * reg.total_bits()
            elif self.adjoint:
                wire_symbols += ["prep†"] * reg.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)
    

    def __repr__(self) -> str:
        return f"LCU Prep"
    

    def decompose_from_registers(
            self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
        ) -> cirq.OP_TREE:

        data = quregs["data"]

        yield cirq.Ry(rads=2*math.atan(self.alpha2/self.alpha1)).on(data[0])
        yield cirq.Ry(rads=2*math.atan(self.alpha0/self.alpha1)).on(data[1])