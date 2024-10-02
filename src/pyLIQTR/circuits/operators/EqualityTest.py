"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
from functools import cached_property
from typing import Set
from numpy.typing import NDArray

from qualtran import (
    QUInt,
    QBit,
    Register,
    Signature,
    GateWithRegisters
)
from qualtran.bloqs.mcmt.and_bloq import MultiAnd, And
from qualtran.bloqs.basic_gates import CNOT

class EqualityTest(GateWithRegisters):
    """
    Implements U|A>|B>|target> = |A>|B>|target^(A == B)>
    """
    def __init__(self,bitsize:int):
        if bitsize >= 2:
            self.bitsize = bitsize
        else:
            raise ValueError(f"EqualityTest bitsize must be greater than 2, found {bitsize}.")

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('A', QUInt(bitsize=self.bitsize)),
                Register('B', QUInt(bitsize=self.bitsize)),
                Register('target', QBit())
            ]
        )

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        A_reg = quregs['A']
        B_reg = quregs['B']
        target = quregs['target']

        ancilla = context.qubit_manager.qalloc(self.bitsize-1) 

        # store bitwise XOR in B
        for a,b in zip(A_reg, B_reg):
            yield CNOT().on(a,b)  
        
        # 0 controlled AND ladder using XOR result. If all XOR result bits are 0, the target of the AND ladder (ancilla[-1]) will be 1, indicating A and B are equal.
        if self.bitsize >= 3:
            and_gate = MultiAnd(cvs=(0,) * self.bitsize)
        else:
            and_gate = And(0, 0)
        
        yield and_gate.on(*B_reg, *ancilla) 

        # update result qubit
        yield CNOT().on(ancilla[-1], *target)  

        # uncompute AND ladder
        yield and_gate.adjoint().on(*B_reg, *ancilla)

        # uncompute XOR
        for a,b in zip(A_reg, B_reg):
            yield CNOT().on(a,b)  

        # free ancilla
        context.qubit_manager.qfree(ancilla)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        if self.bitsize >= 3:
            and_gate = MultiAnd(cvs=(0,) * self.bitsize)
        else:
            and_gate = And(0, 0)
        return {(and_gate, 1),(and_gate.adjoint(),1),(CNOT(),2*self.bitsize+1)}