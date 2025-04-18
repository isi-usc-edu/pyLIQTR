"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
from functools import cached_property
from typing import Set, Optional, Tuple
from numpy.typing import NDArray

from qualtran import (
    QUInt,
    QBit,
    Register,
    Side,
    Signature,
    GateWithRegisters
)
from qualtran.drawing import directional_text_box
from qualtran.bloqs.arithmetic.addition import Add
from pyLIQTR.circuits.operators.ControlledCopy import ControlledCopy

class Product(GateWithRegisters):
    """
    Implements U|A>|B>|0> = |A>|B>|A*B> as described in the reference. To understand the procedure, let

    .. math::

        A = \\sum_{\\ell=1}^{d_A} 2^{d_A-\\ell} A_\\ell

        B = \\sum_{\\ell=1}^{d_B} 2^{d_B-\\ell} B_\\ell

    then we can write the product as

    .. math::
        AB  &= \\sum_{\\ell=1}^{d_A} \\sum_{m=1}^{d_B} 2^{d_A-\\ell} 2^{d_B-m} A_\\ell  B_m \\\\
            &= \\sum_{\\ell=1}^{d_A} 2^{d_A-\\ell} A_\\ell \\left( 2^{d_B-1} B_1 + 2^{d_B-2} B_2 + \\dots + 2^0 B_{d_B} \\right) \\\\
            &= A \\times \\left( 2^{d_B-1} B_1 + 2^{d_B-2} B_2 + \\dots + 2^0 B_{d_B} \\right)

    Then for each factor of :math:`2^{d_B-i}`, we can use the :math:`i`-th bit of B to control a copy of the A register into an ancilla, such that the ancilla contains the state :math:`|B_i A\\rangle`. Then to account for the power of 2, we can add the ancilla into a shifted portion of the target output register. This addition only needs to account for :math:`d_A + 1` bits, where the +1 is for the carry bit.

    Registers:

    .. line-block::
        A: Register holding value A.
        B: Register holding value B. Size must be less than or equal to the size of register A.
        target: Register to output the product A*B

    References:
        `Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization <https://arxiv.org/abs/2007.07391>`_ Appendix D.3

    :param int a_bitsize: Size of register A
    :param Optional[int] b_bitsize: Size of register B. Default is equal to a_bitsize.
    """
    def __init__(self,a_bitsize:int,b_bitsize:Optional[int]=None):
        self.a_bitsize = a_bitsize
        if b_bitsize is None:
            self.b_bitsize = a_bitsize
        else:
            if b_bitsize > a_bitsize:
                raise ValueError(f"b_bitsize={b_bitsize} must be less than or equal to a_bitsize={a_bitsize}")
            self.b_bitsize = b_bitsize

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('A', QUInt(bitsize=self.a_bitsize)),
                Register('B', QUInt(bitsize=self.b_bitsize)),
                Register('target', QUInt(bitsize=self.a_bitsize+self.b_bitsize),side=Side.RIGHT)
            ]
        )

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["In(x)"] * int(self.a_bitsize)
        wire_symbols += ["In(y)"] * int(self.b_bitsize)
        wire_symbols += ["Out(x*y)"] * int(self.a_bitsize+self.b_bitsize)
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':

        if reg is None:
            return Text("")
        if reg.name == 'A':
            return directional_text_box('A', side=reg.side)
        elif reg.name == 'B':
            return directional_text_box('B', side=reg.side)
        elif reg.name == 'target':
            return directional_text_box('A*B', side=reg.side)
        else:
            raise ValueError()

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        A_reg = quregs['A']
        B_reg = quregs['B']
        target = quregs['target']
        ancilla = context.qubit_manager.qalloc(self.a_bitsize)

        copy_gate = ControlledCopy(self.a_bitsize)
        add_gate = Add(a_dtype=QUInt(self.a_bitsize),b_dtype=QUInt(self.a_bitsize+1))

        # copy A controlled on lsb of B for Hamming weight of 1 (2**0)
        yield copy_gate.on_registers(control=B_reg[-1],original=A_reg,target=target[self.b_bitsize:])

        # for Hamming weight of 2**ell, control copy A to ancilla register and then shift by adding to shifted position on target register
        for ell in range(1,self.b_bitsize):
            yield copy_gate.on_registers(control=B_reg[self.b_bitsize-1-ell],original=A_reg,target=ancilla) #-1 since indexing starts at 0
            yield add_gate.on_registers(a=ancilla,b=target[self.b_bitsize-1-ell:-ell])
            yield copy_gate.adjoint().on_registers(control=B_reg[self.b_bitsize-1-ell],original=A_reg,target=ancilla)

        context.qubit_manager.qfree(ancilla)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        copy_gate = ControlledCopy(self.a_bitsize)
        add_gate = Add(a_dtype=QUInt(self.a_bitsize),b_dtype=QUInt(self.a_bitsize+1))
        return {(copy_gate, self.b_bitsize),(copy_gate.adjoint(),self.b_bitsize-1),(add_gate,self.b_bitsize-1)}