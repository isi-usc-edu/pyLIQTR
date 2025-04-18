"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
from functools import cached_property
from numpy.typing import NDArray
from typing import Set
from qualtran import GateWithRegisters, Signature, QAny, QBit, Side, Register
from qualtran.bloqs.mcmt import MultiAnd, And
from qualtran.bloqs.basic_gates import XGate

class FlagZeroAsFailure(GateWithRegisters):
    """Flags the state representing minus zero in the :math:`\\nu` register.

    Parameters:
        bitsize: the number of bits in the :math:`\\nu` register.

    Registers:

    .. line-block::
        nu: The register being checked for the minus zero state.
        flag_dim: Qubits flagging occurence of minus zero for each dimension of :math:`\\nu`.
        flag_minus_zero: Qubit flagging occurence of minus zero overall.
        and_ancilla: Ancilla used for the And ladder.
        flag_ancilla: Ancilla used for OR to mark flag_minus_zero.

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_
        Eq 80, pg 22
    """

    def __init__(self,bitsize:int):
        self.bitsize = bitsize 

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("nu", QAny(self.bitsize + 1), shape=(3,)),
                Register("flag_dim", QBit(), shape=(3,), side=Side.RIGHT),
                Register("flag_minus_zero", QBit(), side=Side.RIGHT),
                Register("and_ancilla", QAny(self.bitsize - 1), shape=(3,), side=Side.RIGHT),
                Register("flag_ancilla", QBit(), side=Side.RIGHT)
            ]
        )

    def pretty_name(self) -> str:
        return r'ν≠-0'

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        nu_reg = quregs['nu']
        flags = quregs['flag_dim']
        ancillas = quregs['and_ancilla']
        flag_minus_zero = quregs['flag_minus_zero'][0]
        flag_ancilla = quregs['flag_ancilla'][0]

        and_gate = MultiAnd(cvs=(1,)+(0,)*self.bitsize)

        # check each dimension for the state |-0> = |10...0> and flag it
        for dim,qbs in enumerate(nu_reg):

            yield and_gate.on_registers(ctrl=[[bit] for bit in qbs],target=flags[dim],junk=[[bit] for bit in ancillas[dim]])

        # check if any of the 3 flag qubits are 1 using 2 OR gates
        yield And(cv1=0,cv2=0).on(flags[0][0],flags[1][0],flag_ancilla)
        yield XGate().on(flag_ancilla)
        yield And(cv1=0,cv2=0).on(flag_ancilla,flags[2][0],flag_minus_zero)
        yield XGate().on(flag_minus_zero)


    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(MultiAnd(cvs=(1,)+(0,)*self.bitsize), 3), (And(cv1=0,cv2=0), 2),(XGate(),2)}
