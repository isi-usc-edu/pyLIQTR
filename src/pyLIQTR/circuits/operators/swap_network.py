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

import math
import cirq
import numpy as np
from typing import Tuple, Sequence
from cirq._compat import cached_property
from qualtran import Signature, GateWithRegisters, Register, SelectionRegister
from qualtran.bloqs.multi_control_multi_target_pauli import MultiControlPauli as mcmtp
from qualtran.bloqs.unary_iteration_bloq import UnaryIterationGate
from pyLIQTR.utils.global_ancilla_manager import GlobalQubitManager as manager

class SwapUnary(UnaryIterationGate):
    """
    Implements
    __________
        Helper Class for SwapNetwork

    Parameters
    __________
        n : int
            2**n = N - size of F1 in Carleman linearization
        K : int
            number of truncation steps in Carleman linearization
        plus : int
            slight modification of accumulator
        control : int
            number of controls
        controlled : bool
            whether or not to control entire unitary
    """

    def __init__(self, n : int, K : int, plus : bool, control : int, controlled : bool):

        self.controlled = controlled
        self.control = control
        self.plus = plus

        self.n = n
        self.K = K

    
    @cached_property
    def vector(self) -> Tuple[int]:
        return (self.n, self.K)
    

    @cached_property
    def control_registers(self) -> Tuple[Register]:
        registers = () if not self.controlled else (Register('control', self.control),)
        return registers
    

    @cached_property
    def selection_registers(self) -> Tuple[SelectionRegister,...]:
        return (SelectionRegister("selection", bitsize = math.ceil(np.log2(self.K))),)
    

    @cached_property
    def target_registers(self) -> Tuple[Register,...]:
        if not self.plus:
            return (Register("target0", bitsize = self.n*self.K), Register("target1", bitsize = 1))
        elif self.plus:
            return (Register("target0", bitsize = self.n*(self.K-1)), Register("target1", bitsize = 1))


    def nth_operation(self, control : Sequence[cirq.Qid], selection : int, context: cirq.DecompositionContext = None, **quregs) -> cirq.OP_TREE:
        
        target0 = quregs["target0"]
        target1 = quregs["target1"]

        if context is None:
            context = manager()
        
        if not self.plus:
            max = self.vector[1]
        elif self.plus:
            max = self.vector[1] - 1

        yield cirq.CX.on(control, target1[0])
        if selection > 0 and selection < max:
            for idx in range(self.n):
                yield cirq.CX.on(target0[idx+selection*self.n-self.n], target0[-1-idx])
                yield mcmtp((1,1), target_gate=cirq.X).on_registers(target=target0[idx+selection*self.n-self.n], controls=[target1[0],target0[-1-idx]])
                yield cirq.CX.on(target0[idx+selection*self.n-self.n], target0[-1-idx])
            

class SwapNetwork(GateWithRegisters):
    """
    Implements
    __________
        Swapping F tensor products to correct tensor level of direct sum
        (TODO: verify with lemma six in paper under development)

    Parameters
    __________
        n : int
            2**n = N - size of F1 in Carleman linearization
        K : int
            number of truncation steps in Carleman linearization
        control : int
            number of controls
        plus : int
            slight modification of accumulator
    """

    def __init__(self, n : int, K : int, control : int, plus : bool):

        self.control = control
        self.plus = plus
        self.n = n
        self.K = K


    @cached_property
    def vector(self) -> Tuple[int]:
        return (self.n, self.K)
    

    @cached_property # don't forget to cache this
    def signature(self) -> Signature:
        if self.plus:
            return Signature.build(data = self.vector[0]*(self.vector[1]-1)+1,
                                   ancilla = math.ceil(np.log2(self.vector[1])), control = self.control)
        elif not self.plus:
            return Signature.build(data = self.vector[0]*self.vector[1]+1,
                                   ancilla = math.ceil(np.log2(self.vector[1])), control = self.control)


    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.signature:
            if reg.name == "control":
                wire_symbols += ["control"] * reg.total_bits()
            elif reg.name == "ancilla":
                wire_symbols += ["ancilla"] * reg.total_bits()
            else:
                if self.plus:
                    wire_symbols += ["SwapNetwork+"] * reg.total_bits()
                elif not self.plus:
                    wire_symbols += ["SwapNetwork"] * reg.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)


    def __repr__(self) -> str:
        return f"Swap Network"


    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:

        data = quregs["data"]
        ancilla = quregs["ancilla"]

        if self.control > 0:
            control = quregs["control"]
            controlled = True
        elif self.control == 0:
            control = []
            controlled = False

        yield SwapUnary(self.vector[0], self.vector[1],
                        self.plus, self.control, controlled).on_registers(selection = ancilla, target0 = data[1:],
                                                               target1 = data[0], control = control)