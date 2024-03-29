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
from typing import Sequence, Union, Tuple, Optional
from numpy.typing import NDArray

import attr
import cirq
import numpy as np
import qualtran as qt

from cirq._compat import cached_property
from qualtran import _infra
from qualtran._infra.registers import SelectionRegister, Register
from qualtran.bloqs.unary_iteration_bloq import UnaryIterationGate

from pyLIQTR.utils.circuit_decomposition import circuit_decompose_multi

@attr.frozen
class BinaryToUnary(UnaryIterationGate):
    """Implements U s.t. U|l>|Psi> -> |l> X_{l - 1} ... X_{0} |Psi>

    where |l> is the ⎡logL⎤-qubit selection register and |Psi> is the L-qubit target register. 
    The gate is implemented using an accumulator bit in the unary iteration circuit as explained
    in the reference below.

    Example, for L = 4:
        l=0: U|00>|0000> -> |00>|0000>
        l=1: U|01>|0000> -> |01>|1000>
        l=2: U|10>|0000> -> |10>|1100>
        l=3: U|11>|0000> -> |11>|1110>

    Args:
        selection_regs: Indexing `select` signature of type `SelectionRegister`. It also contains
            information about the iteration length of each selection register.
        controlled: Set to True for controlled operation

    References:
        See Fig 8 of https://arxiv.org/abs/1805.03662 for more details.
    """

    selection_regs: Tuple[SelectionRegister, ...] = attr.field(
        converter=lambda v: (v,) if isinstance(v, SelectionRegister) else tuple(v)
    )
    controlled: Optional[bool] = False

    @classmethod
    def make_on(
        cls,
        **quregs: Union[Sequence[cirq.Qid], NDArray[cirq.Qid]],  # type: ignore[type-var]
    ) -> cirq.Operation:
        """Helper constructor to automatically deduce selection_regs attribute."""
        controlled = True if quregs.get('control', 0) else False
        return BinaryToUnary(
            selection_regs=SelectionRegister(
                'selection', len(quregs['selection']), len(quregs['target'])
            ),
            controlled = controlled
        ).on_registers(**quregs)

    @cached_property
    def control_registers(self) -> Tuple[Register]:
        registers = () if not self.controlled else (Register('control', 1),)
        return registers

    @cached_property
    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        return self.selection_regs

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        total_iteration_size = np.prod(
            tuple(reg.iteration_length for reg in self.selection_registers)
        )
        return (Register('target', int(total_iteration_size)),)

    @cached_property
    def extra_registers(self) -> Tuple[Register, ...]:
        return (Register('accumulator', 1),)

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        quregs['accumulator'] = np.array(context.qubit_manager.qalloc(1))
        control = quregs.get('control',())
        yield cirq.X(*quregs['accumulator']).controlled_by(*control)
        yield super(BinaryToUnary, self).decompose_from_registers(
            context=context, **quregs
        )
        context.qubit_manager.qfree(quregs['accumulator'])

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@"] * _infra.gate_with_registers.total_bits(self.control_registers)
        wire_symbols += ["In"] * _infra.gate_with_registers.total_bits(self.selection_registers)
        wire_symbols += [f"X"] * _infra.gate_with_registers.total_bits(self.target_registers)
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def nth_operation(  # type: ignore[override]
        self,
        context: cirq.DecompositionContext,
        control: cirq.Qid,
        target: Sequence[cirq.Qid],
        accumulator: Sequence[cirq.Qid],
        **selection_indices: int,
    ) -> cirq.OP_TREE:
        selection_shape = tuple(reg.iteration_length for reg in self.selection_regs)
        selection_idx = tuple(selection_indices[reg.name] for reg in self.selection_regs)
        target_idx = int(np.ravel_multi_index(selection_idx, selection_shape))
        yield cirq.CNOT(control, *accumulator)
        yield cirq.CNOT(*accumulator, target[target_idx])

@attr.frozen
class BinaryToUnaryBits(UnaryIterationGate):
    """
    Implements U s.t. 

    U|l>|Psi> -> |l> X_{k - 1} ... X_{0} |Psi>

    where |l> is the L-qubit selection register, |Psi> is the L-qubit target register, l is the selection index in the range [0,2**L-1], k = ⎡log2(l+1)⎤, and X_{i} indicates the Pauli X gate applied to the ith qubit.
    This is step 3(a)i on page 52 of ref [1] -- "produce a new L-qubit register that has zeros matching the leading zeros in the binary representation of l, and ones after that"
    The gate is implemented using an accumulator bit in the unary iteration circuit as explained in reference [2].

    Example, for L = 3:

        l=0: U|000>|000> -> |000>|000> (three leading zeros)
        l=1: U|001>|000> -> |001>|100> (two leading zeros)
        l=2: U|010>|000> -> |010>|110> (one leading zero)
        l=3: U|011>|000> -> |011>|110> (one leading zero)
        l=4: U|100>|000> -> |100>|111> (no leading zeros)
        ...
        NOTE: selection register has msb on left (first) but target register has msb on right (last)

    Args:
        selection_regs: Indexing `select` signature of type `SelectionRegister`. It also contains
            information about the iteration length of each selection register.
        controlled: Set to True for controlled operation

    References:
        [1] https://arxiv.org/abs/2011.03494 
        [2] See Fig 8 of https://arxiv.org/abs/1805.03662 for details on accumulator bit usage.
    """

    selection_regs: Tuple[SelectionRegister, ...] = attr.field(
        converter=lambda v: (v,) if isinstance(v, SelectionRegister) else tuple(v)
    )
    controlled: Optional[bool] = False

    @classmethod
    def make_on(
        cls,
        **quregs: Union[Sequence[cirq.Qid], NDArray[cirq.Qid]],  # type: ignore[type-var]
    ) -> cirq.Operation:
        """Helper constructor to automatically deduce selection_regs attribute."""
        controlled = True if quregs.get('control', 0) else False
        n_bits = len(quregs['selection'])
        iteration_length = 2**n_bits
        return BinaryToUnaryBits(
            selection_regs=SelectionRegister(
                'selection', len(quregs['selection']), iteration_length
            ),
            controlled = controlled
        ).on_registers(**quregs)

    @cached_property
    def control_registers(self) -> Tuple[Register]:
        registers = () if not self.controlled else (Register('control', 1),)
        return registers

    @cached_property
    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        return self.selection_regs

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        target_bitsize = self.selection_registers[0].bitsize
        return (Register('target', target_bitsize),)

    @cached_property
    def extra_registers(self) -> Tuple[Register, ...]:
        return (Register('accumulator', 1),)

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        quregs['accumulator'] = np.array(context.qubit_manager.qalloc(1))
        control = quregs.get('control',())
        yield cirq.X(*quregs['accumulator']).controlled_by(*control)
        yield super(BinaryToUnaryBits, self).decompose_from_registers(
            context=context, **quregs
        )
        context.qubit_manager.qfree(quregs['accumulator'])

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@"] * _infra.gate_with_registers.total_bits(self.control_registers)
        wire_symbols += ["In"] * _infra.gate_with_registers.total_bits(self.selection_registers)
        wire_symbols += [f"X"] * _infra.gate_with_registers.total_bits(self.target_registers)
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def nth_operation(  # type: ignore[override]
        self,
        context: cirq.DecompositionContext,
        control: cirq.Qid,
        target: Sequence[cirq.Qid],
        accumulator: Sequence[cirq.Qid],
        **selection_indices: int,
    ) -> cirq.OP_TREE:
        selection_shape = tuple(reg.iteration_length for reg in self.selection_regs)
        selection_idx = tuple(selection_indices[reg.name] for reg in self.selection_regs)
        total_selection_idx = int(np.ravel_multi_index(selection_idx, selection_shape))
        # if total_selection_index is a power of 2, apply CNOT to target
        if (total_selection_idx and (not(total_selection_idx & (total_selection_idx - 1))) ): 
            target_idx = int(np.log2(total_selection_idx))
            yield cirq.CNOT(*accumulator, target[target_idx])
        # CNOT to turn off accumulator if control is 1
        yield cirq.CNOT(control, *accumulator)