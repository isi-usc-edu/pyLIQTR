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

from cirq._compat import cached_property
from qualtran._infra.registers import SelectionRegister, Register, Signature
from qualtran._infra.gate_with_registers import GateWithRegisters, total_bits
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.bloqs.unary_iteration_bloq import UnaryIterationGate
from qualtran.bloqs.and_bloq import And

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
        **quregs: Union[Sequence[cirq.Qid], NDArray[cirq.Qid]],
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
        wire_symbols = ["@"] * total_bits(self.control_registers)
        wire_symbols += ["In"] * total_bits(self.selection_registers)
        wire_symbols += [f"X"] * total_bits(self.target_registers)
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def nth_operation( 
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
class BinaryToUnaryBits(GateWithRegisters):
    """
    Implements U s.t. 

    U|l>|Psi> -> |l> X_{k - 1} ... X_{0} |Psi>

    where |l> is the L-qubit selection register, |Psi> is the L-qubit target register, l is the selection index in the range [0,2**L-1], k = ⎡log2(l+1)⎤, and X_{i} indicates the Pauli X gate applied to the ith qubit.
    This is step 3(a)i on page 52 of ref [1] -- "produce a new L-qubit register that has zeros matching the leading zeros in the binary representation of l, and ones after that"

    NOTE: This construction assumes the unary register is initialized to the |0> state and will return to the |0> state when inverted.

    Example, for L = 3:

        l=0: U|000>|000> -> |000>|000> (three leading zeros)
        l=1: U|001>|000> -> |001>|001> (two leading zeros)
        l=2: U|010>|000> -> |010>|011> (one leading zero)
        l=3: U|011>|000> -> |011>|011> (one leading zero)
        l=4: U|100>|000> -> |100>|111> (no leading zeros)
        ...

    Args:
        n_bits: The number of qubits in the binary and unary registers

    References:
        [1] https://arxiv.org/abs/2011.03494 
    """

    n_bits: int

    @classmethod
    def make_on(
        cls,
        **quregs: Union[Sequence[cirq.Qid], NDArray[cirq.Qid]],
    ) -> cirq.Operation:
        """Helper constructor to automatically deduce n_bits based on registers."""
        return BinaryToUnaryBits(
            n_bits=len(quregs['binary'])
        ).on_registers(**quregs)

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(binary=self.n_bits,unary=self.n_bits)

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        binary, unary = quregs['binary'], quregs['unary']

        open_and_gate = And(cv1=0,cv2=0)

        # copy the msb from the binary register to the unary register
        yield cirq.CNOT(binary[0],unary[0])

        # do (n_bits-1) OR operations such that unary[n] = (binary[n] OR unary[n-1]) for n=[1,n_bits-1], where n=0 is the msb.
        for n in range(1,self.n_bits):
            yield open_and_gate.on_registers(ctrl=[[binary[n]],[unary[n-1]]], target=unary[n])
            yield cirq.X(unary[n])

    def _t_complexity_(self) -> TComplexity:
        num_ands = self.n_bits-1
        resources_per_and = And(cv1=0,cv2=0)._t_complexity_()
        return TComplexity(t=resources_per_and.t*num_ands,clifford=resources_per_and.clifford*num_ands+self.n_bits)