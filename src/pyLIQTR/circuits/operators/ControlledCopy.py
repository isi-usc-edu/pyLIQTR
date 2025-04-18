"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
import numpy as np
from functools import cached_property
from typing import Set, Optional, Tuple, Sequence
from numpy.typing import NDArray
from attrs import frozen

from qualtran import (
    QUInt,
    QBit,
    BoundedQUInt,
    Register,
    Side,
    Signature,
    GateWithRegisters
)
from qualtran.bloqs.basic_gates import XGate
from qualtran.bloqs.mcmt import And
from qualtran.bloqs.multiplexers.unary_iteration_bloq import UnaryIterationGate
from qualtran._infra.gate_with_registers import total_bits

@frozen
class ControlledCopy(GateWithRegisters):
    """
    Implements U|ctrl>|original>|0> = |ctrl>|original>|original*ctrl>. This can also be thought of as an out of place multiplication of |original> by the bit |ctrl>. Note, this construction assumes the target state is initialized to |0>.

    Registers:

    .. line-block::
        control: Register to control on.
        target: Register the copy will be output on.
        original: Register to copy.

    :param int bitsize: Size of the register to be copied (original)
    :param Optional[int] control_val: Value to control on.
    """

    bitsize: int
    control_val: Optional[int] = 1

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('control', QBit()),
                Register('original', QUInt(bitsize=self.bitsize)),
                Register('target', QUInt(bitsize=self.bitsize))
            ]
        )

    def __repr__(self):
        return f'ControlledCopy(bitsize={self.bitsize}, control_val={self.control_val})'

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        original = quregs['original']
        control = quregs['control']
        target = quregs['target']

        if self.control_val == 0:
            yield XGate().on(*control)

        for i,bit in enumerate(original):
            yield And().on_registers(ctrl=[control,[bit]],target=target[i])

        if self.control_val == 0:
            yield XGate().on(*control)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        if self.control_val == 0:
            return {(And(), self.bitsize),(XGate(),2)}
        return {(And(), self.bitsize)}

class MultiplexedControlledCopy(UnaryIterationGate):
    """
    A multiplexed version of :class:`ControlledCopy`. When the selection register stores index i, original[i] is copied to the target register. 

    Registers:

    .. line-block::
        control: Register to control unary iteration on if controlled=True.
        selection: Selection register.
        target: Register the copy will be output on.
        original: Register to copy. Should have shape=(target_bitsize,iteration_length).

    :param int target_bitsize: Size of the register to copy onto.
    :param int selection_bitsize: Size of selection register. Default is 2.
    :param int iteration_length: Number of items to iterate over. Default is 3.
    :param bool controlled: Set to True if the unary iteration should be controlled on a control qubit. Default is True.
    """

    def __init__(self,target_bitsize,selection_bitsize=2,iteration_length=3,controlled=True):
        self.selection_bitsize=selection_bitsize
        self.target_bitsize = target_bitsize
        self.iteration_length = iteration_length
        self.controlled = controlled

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return (Register('control',dtype=QBit()),) if self.controlled else ()

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (Register(
                'selection', dtype=BoundedQUInt(bitsize=self.selection_bitsize,iteration_length=self.iteration_length) # w 
            ),)

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (Register('target', QUInt(bitsize=self.target_bitsize)),) # ancilla

    @cached_property
    def extra_registers(self) -> Tuple[Register, ...]:
        return (Register('original', QUInt(bitsize=self.target_bitsize),shape=(self.iteration_length,)),) # sys[eta]

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [*self.control_registers,*self.selection_registers, *self.target_registers,*self.extra_registers]
        )

    def nth_operation( 
        self,
        context: cirq.DecompositionContext,
        control: cirq.Qid,
        target: Sequence[cirq.Qid],
        original: Sequence[cirq.Qid],
        **selection_indices: int,
    ) -> cirq.OP_TREE:
        selection_shape = tuple(reg.dtype.iteration_length for reg in self.selection_registers)
        selection_idx = tuple(selection_indices[reg.name] for reg in self.selection_registers)
        target_idx = int(np.ravel_multi_index(selection_idx, selection_shape))
        yield ControlledCopy(bitsize=self.target_bitsize).on_registers(control=control,original=original[target_idx],target=target)

    def nth_operation_callgraph(self, **selection_regs_name_to_val) -> Set['BloqCountT']:
        return {(ControlledCopy(bitsize=self.target_bitsize),1)}

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@"] * total_bits(self.control_registers)
        wire_symbols += ["In"] * total_bits(self.selection_registers)
        wire_symbols += ["copy"] * total_bits(self.target_registers)
        wire_symbols += ["original"] * total_bits(self.extra_registers)
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)