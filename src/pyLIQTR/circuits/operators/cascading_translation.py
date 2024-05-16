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
from typing import Tuple, Sequence
from cirq._compat import cached_property
from qualtran import Signature, GateWithRegisters, Register, SelectionRegister
from pyLIQTR.utils.global_ancilla_manager import GlobalQubitManager as manager

from qualtran.bloqs.and_bloq import And 
from pyLIQTR.circuits.operators.translation import Translation
from qualtran.bloqs.unary_iteration_bloq import UnaryIterationGate

class CascadingUnary(UnaryIterationGate):
    """
    Implements
    __________
        Helper Class for CascadingTranslation

    Parameters
    __________
        n : int
            2**n = N - size of F1 in Carleman linearization
        K : int
            number of truncation steps in Carleman linearization
        direction : str
            left or right
        optimize : bool
            passed to subfunctions for lower T count and depth
        controlled : bool
            whether or not to control entire unitary
    """

    def __init__(self, n : int, K : int, direction : str, optimize : bool, controlled : bool = False):

        self.controlled = controlled 
        self.direction = direction
        self.optimize = optimize
        self.n = n
        self.K = K

    
    @cached_property
    def vector(self) -> Tuple[int]:
        return (self.n, self.K)
    

    @cached_property
    def control_registers(self) -> Tuple[Register]:
        registers = () if not self.controlled else (Register('control', 1),)
        return registers
    

    @cached_property
    def selection_registers(self) -> Tuple[SelectionRegister,...]:
        return (SelectionRegister("selection", bitsize = math.ceil(np.log2(self.K))),)
    

    @cached_property
    def target_registers(self) -> Tuple[Register,...]:
        return (Register("target0", bitsize = self.n*self.K+2), Register("target1", bitsize = 1))


    def nth_operation(self, selection : int, control : cirq.Qid, context: cirq.DecompositionContext = None, **quregs) -> cirq.OP_TREE:
        
        target0 = quregs["target0"]
        target1 = quregs["target1"]

        if context is None:
            context = manager()

        yield cirq.CX.on(control, target1[0])
        if selection > 0 and selection < self.vector[1]:
            yield Translation(self.vector[0]*self.vector[1]+2, (self.vector[0]*selection, False),
                              self.direction, (True, 1), self.optimize).on_registers(data=target0, control=target1[0])


class CascadingTranslation(GateWithRegisters):
    """
    Implements
    __________
        A cascade of translations by 0,N,N+N^2,N+N^2+N^3,...,\prod_{j=0}^{K-1}N^j 

    Parameters
    __________
        n : int
            2**n = N - size of F1 in Carleman linearization
        K : int
            number of truncation steps in Carleman linearization
        direction : str
            left or right
        optimize : bool
            passed to subfunctions for lower T count and depth

    Raises
    ______
        valueError
            If K < 2
            If n < 1
    """

    def __init__(self, n: int, K: int, direction : str, optimize : bool):

        if not ((direction == "left") or (direction == "right")):
            if not ((direction == "l") or (direction == "r")):
                raise ValueError("direction must be l(eft) or r(ight)")

        if K < 2 or n < 1:
            raise ValueError("n must be positive and K must be greated than one")
        
        self.direction = direction
        self.optimize = optimize

        self.n = n
        self.K = K


    @cached_property
    def vector(self) -> Tuple[int]:
        return (self.n, self.K)
    

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(ancilla = 1 + math.ceil(np.log2(self.vector[1])), 
                               data = self.vector[0] * self.vector[1] + 2)
        

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.signature:
            if reg.name == "ancilla":
                wire_symbols += ["ancilla"] * reg.total_bits()
            else:
                wire_symbols += ["CascadingTranslation"] * reg.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)
    

    def __repr__(self) -> str:
        return f"Cascading Translation"
    

    def decompose_from_registers(
            self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
        ) -> cirq.OP_TREE:

        data = quregs["data"]
        ancilla = quregs["ancilla"]

        yield CascadingUnary(self.vector[0], self.vector[1], 
                             self.direction, self.optimize).on_registers(selection = ancilla[1:], target0 = data, target1 = ancilla[0])
        yield cirq.X.on(ancilla[0])