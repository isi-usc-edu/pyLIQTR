"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
from functools import cached_property
from typing import Set, Optional, Union, Sequence, Callable, Dict, Tuple
from numpy.typing import NDArray

from qualtran import GateWithRegisters, Signature, Register, QBit, QAny, BoundedQUInt
from qualtran.bloqs.basic_gates import Toffoli, ZGate, Hadamard, CZPowGate
from qualtran.bloqs.bookkeeping import ArbitraryClifford
from qualtran.cirq_interop import CirqGateAsBloq
from qualtran.bloqs.mcmt import MultiControlPauli

from pyLIQTR.circuits.operators.ControlledCopy import MultiplexedControlledCopy

class SelectT_FirstQuantized(GateWithRegisters):
    """SELECT for the kinetic energy operator for the first quantized chemistry Hamiltonian.

    Registers:

    .. line-block::
        sys: The system register.
        plus: A :math:`|+\\rangle` state.
        flag_T: a flag to control on the success of the :math:`T` state preparation.
        w: Register used to select between the 3 dimensions of the system register.
        r: Register used to select bit of system register.
        s: Register used to select bit of system register.

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_
        section II.B, pg 20 steps 1-5

    :param int num_bits_p: The number of bits to represent each dimension of the system register.
    """

    def __init__(self,num_bits_p:int):

        self.num_bits_p = num_bits_p

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("flag_T", QBit()),
                Register("plus", QBit()),
                Register("w", BoundedQUInt(bitsize=2,iteration_length=3)),
                Register("r", QAny(bitsize=self.num_bits_p)),
                Register("s", QAny(bitsize=self.num_bits_p)),
                Register("sys", QAny(bitsize=self.num_bits_p), shape=(3,)),
            ]
        )

    def pretty_name(self) -> str:
        return r'SEL T'

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        w = quregs['w']
        r = quregs['r']
        s = quregs['s']
        control = quregs['flag_T']
        plus = quregs['plus']
        p_3d = quregs['sys']
        ancilla = context.qubit_manager.qalloc(self.num_bits_p+2)
        p_ancilla = ancilla[2:]
        pr_target = ancilla[0]
        ps_target = ancilla[1]

        # copy w'th component of p onto ancilla register, controlled on flag_T
        mcc = MultiplexedControlledCopy(target_bitsize=self.num_bits_p)
        yield mcc.on_registers(control=control,selection=w,target=p_ancilla,original=p_3d)

        # |r> and |s> are one-hot unary so just need loop of Toffolis to copy bits pr and ps to ancilla bits
        for i in range(self.num_bits_p):
            yield Toffoli().on_registers(ctrl=[[r[i]],[p_ancilla[i]]], target=pr_target)
            yield Toffoli().on_registers(ctrl=[[s[i]],[p_ancilla[i]]], target=ps_target)

        # controlled phase *unless* both bits are 1
        yield ZGate().on(*plus)
        yield MultiControlPauli(cvs=(1,1),target_gate=cirq.Z).on_registers(controls=[[pr_target],[ps_target]],target=plus)

        # erase ancilla using measurement and phase fixup. We measure the target bit once and then use classically controlled CZs on each set of controls for the phase fixup.
        ## for ps and pr
        yield Hadamard().on(ps_target)
        yield cirq.measure(ps_target, key=f"{ps_target}")
        yield Hadamard().on(pr_target)
        yield cirq.measure(pr_target, key=f"{pr_target}")
        for i in range(self.num_bits_p):
            yield cirq.CZ.on(s[i],p_ancilla[i]).with_classical_controls(f"{ps_target}")
            yield cirq.CZ.on(r[i],p_ancilla[i]).with_classical_controls(f"{pr_target}")
        yield cirq.reset(ps_target)
        yield cirq.reset(pr_target)
        ## for pw
        yield mcc.adjoint().on_registers(control=control,selection=w,target=p_ancilla,original=p_3d)
        context.qubit_manager.qfree(ancilla)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # use ArbitraryClifford(n=1) for cirq.reset
        mcc = MultiplexedControlledCopy(target_bitsize=self.num_bits_p)
        return {(mcc, 1),(MultiControlPauli(cvs=(1,1),target_gate=cirq.Z),1), (Toffoli(),2*self.num_bits_p),(ZGate(),1),(Hadamard(),2),(CirqGateAsBloq(cirq.MeasurementGate(num_qubits=1)),2),(cirq.ClassicallyControlledOperation(sub_operation=cirq.CZ(cirq.LineQubit(0),cirq.LineQubit(1)),conditions=''),2*self.num_bits_p),(ArbitraryClifford(n=1),2),(mcc.adjoint(),1)}

    def call_graph(
        self,
        generalizer: Optional[Union['GeneralizerT', Sequence['GeneralizerT']]] = None,
        keep: Optional[Callable[['Bloq'], bool]] = None,
        max_depth: Optional[int] = None,
    ) -> Tuple['nx.DiGraph', Dict['Bloq', Union[int, 'sympy.Expr']]]:
        """Get the bloq call graph and call totals.

        The call graph has edges from a parent bloq to each of the bloqs that it calls in
        its decomposition. The number of times it is called is stored as an edge attribute.
        To specify the bloq call counts for a specific node, override `Bloq.build_call_graph()`.

        Args:
            generalizer: If provided, run this function on each (sub)bloq to replace attributes
                that do not affect resource estimates with generic sympy symbols. If the function
                returns `None`, the bloq is omitted from the counts graph. If a sequence of
                generalizers is provided, each generalizer will be run in order.
            keep: If this function evaluates to True for the current bloq, keep the bloq as a leaf
                node in the call graph instead of recursing into it.
            max_depth: If provided, build a call graph with at most this many layers.

        Returns:
            g: A directed graph where nodes are (generalized) bloqs and edge attribute 'n' reports
                the number of times successor bloq is called via its predecessor.
            sigma: Call totals for "leaf" bloqs. We keep a bloq as a leaf in the call graph
                according to `keep` and `max_depth` (if provided) or if a bloq cannot be
                decomposed.
        """
        from qualtran.resource_counting import get_bloq_call_graph

        if keep is None:
            # we treat measurements and classically controlled operations as leaf nodes
            def keep_measurements(op):
                if isinstance(op,cirq.MeasurementGate):
                    return True
                elif isinstance(op,cirq.ClassicallyControlledOperation):
                    return True
                return False

            keep = keep_measurements

        return get_bloq_call_graph(self, generalizer=generalizer, keep=keep, max_depth=max_depth)
