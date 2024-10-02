"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
from functools import cached_property
from typing import Set
from numpy.typing import NDArray

from qualtran import (
    GateWithRegisters,
    BoundedQUInt,
    QBit,
    Register,
    Signature,
    Side,
)

from pyLIQTR.circuits.operators.FlaggedPrepareUniformSuperposition import FlaggedPrepareUniformSuperposition
from pyLIQTR.circuits.operators.EqualityTest import EqualityTest

class UniformSuperpositionIJFirstQuantization(GateWithRegisters):
    """Prepares a uniform superposition over :math:`$\\eta$` values of :math:`i` and :math:`j` in binary such that :math:`i \\neq j`. The resulting state is

    .. math::
        \\frac{1}{\\sqrt{\\eta}} \\left( \\sqrt{\\eta - 1}|0\\rangle\sum_{i\\neq j=1}^{\\eta}|i\\rangle|j\\rangle + |1\\rangle\sum_{j=1}^{\\eta}|j\\rangle|j\\rangle  \\right)

    Registers:
        i: a :math:`n_\\eta` bit register for binary encoding of :math:`$\\eta$` numbers.
        j: a :math:`n_\\eta` bit register for binary encoding of :math:`$\\eta$` numbers.
        succ_i: Qubit flagging success of the uniform superposition preparation for register i.
        succ_j: Qubit flagging success of the uniform superposition preparation for register j.
        flag_equal: Qubit flagging the states where :math:`i=j`.
        less_than_ancilla: Qubits used for inequality test in uniform superposition preparation.
        rotation_ancilla: Qubits rotated for amplitude amplification.

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_
        section II.A, above Eq 62, pg 18.

    :param int eta: The number of states in the uniform superposition. Equal to the number of electrons in the reference.
    """
    def __init__(self,eta:int):
        self.eta = eta
        self.n_eta = (eta - 1).bit_length()

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('i', BoundedQUInt(bitsize=self.n_eta, iteration_length=self.eta)),
                Register('j', BoundedQUInt(bitsize=self.n_eta, iteration_length=self.eta)),
                Register('succ_i', QBit(), side=Side.RIGHT),
                Register('succ_j', QBit(), side=Side.RIGHT),
                Register('flag_equal', QBit(), side=Side.RIGHT),
                Register('less_than_ancilla', QBit(), shape=(2,1), side=Side.RIGHT),
                Register('rotation_ancilla', QBit(), shape=(2,1), side=Side.RIGHT),
            ]
        )

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        i_reg = quregs['i']
        j_reg = quregs['j']
        success_i = quregs['succ_i']
        success_j = quregs['succ_j']
        flag_equal = quregs['flag_equal']
        less_than_ancilla = quregs['less_than_ancilla']
        rotation_ancilla = quregs['rotation_ancilla']

        uni_sup_gate = FlaggedPrepareUniformSuperposition(d=self.eta)

        yield uni_sup_gate.on_registers(target=i_reg,less_than_ancilla=less_than_ancilla[0][0],rot_ancilla=rotation_ancilla[0][0],success=success_i)

        yield uni_sup_gate.on_registers(target=j_reg,less_than_ancilla=less_than_ancilla[1][0],rot_ancilla=rotation_ancilla[1][0],success=success_j)

        yield EqualityTest(bitsize=self.n_eta).on_registers(A=i_reg,B=j_reg,target=flag_equal)


    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(FlaggedPrepareUniformSuperposition(d=self.eta),2),(EqualityTest(bitsize=self.n_eta),1)}