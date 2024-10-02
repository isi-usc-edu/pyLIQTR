"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
from functools import cached_property
from numpy.typing import NDArray
from typing import Dict, Set
from qualtran import GateWithRegisters, Signature
from qualtran.bloqs.mcmt import And
import cirq

from qualtran.bloqs.basic_gates import Hadamard, TGate, SGate, XGate, CNOT

class CatalyticallyControlledHadamard(GateWithRegisters):
    """Implements a Hadamard controlled using a catalytic state |T>=T|+>. This construction uses only one And gate, which requres 4 T gates, and Clifford gates. There is one additional T gate used to prepare the catalytic state, which is a one time cost since the state can be reused.

    :param bool prepare_catalytic_state: Flag for preparing the catalytic state from |0>. Set to True if the catalytic state has not already been prepared.

    Registers:
        control: The control qubit.
        target: The target for the Hadamard gate.
        catalytic: The catalytic state.

    References:
        `Even more efficient quantum computations of chemistry through tensor hypercontraction <https://arxiv.org/abs/2011.03494>`_
        Figure 17
    """

    def __init__(self,prepare_catalytic_state:bool=False):
        self.prepare_catalytic_state = prepare_catalytic_state 

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(control=1,target=1,catalytic=1)

    def pretty_name(self) -> str:
        return 'CH'

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        control, target, catalytic_state = quregs['control'][0], quregs['target'][0], quregs['catalytic'][0]

        if self.prepare_catalytic_state:
            yield Hadamard().on(catalytic_state)
            yield TGate().on(catalytic_state)

        ancilla = context.qubit_manager.qalloc(1)

        yield Hadamard().on(target)
        yield SGate().on(target)
        yield Hadamard().on(target)

        yield XGate().on(target)
        yield CNOT().on(target,catalytic_state)
        yield XGate().on(target)

        yield And(cv1=0,cv2=1).on(control,catalytic_state,*ancilla)
        yield CNOT().on(target,*ancilla)
        yield SGate().on(*ancilla)
        yield CNOT().on(target,*ancilla)
        yield And(cv1=0,cv2=1,uncompute=True).on(control,catalytic_state,*ancilla)

        yield CNOT().on(control,target)
        yield CNOT().on(target, catalytic_state)

        yield Hadamard().on(target)
        yield SGate().on(target)**(-1)
        yield Hadamard().on(target)

        context.qubit_manager.qfree(ancilla)