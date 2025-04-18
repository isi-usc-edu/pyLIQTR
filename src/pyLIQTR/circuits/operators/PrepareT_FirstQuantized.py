"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
from functools import cached_property
from typing import Set
from numpy.typing import NDArray

from qualtran import GateWithRegisters, Signature, Register, BoundedQUInt, QBit, Side

from pyLIQTR.circuits.operators.PreparePowerTwoState import PreparePowerTwoState
from pyLIQTR.circuits.operators.FlaggedPrepareUniformSuperposition import FlaggedPrepareUniformSuperposition


class PrepareT_FirstQuantized(GateWithRegisters):
    """PREPARE for the kinetic energy operator for the first quantized chemistry Hamiltonian.

    This prepares the state

    .. math::
        \\sum_{w=0}^{2}|w\\rangle
        \\sum_{r=0}^{n_{p}-2}2^{r/2}|r\\rangle
        \\sum_{s=0}^{n_{p}-2}2^{s/2}|s\\rangle

    Registers:

    .. line-block::
        w: a register to index one of three components of the momenta.
        r: a register encoding bits for each component of the momenta.
        s: a register encoding bits for each component of the momenta.
        less_than_ancilla: ancilla used for the uniform state preparation.
        rot_ancilla: ancilla used for the uniform state preparation.
        catalytic: The catalytic state for controlled Hadamards.
        flag_w: A qubit flagging the succes of the preparation of the w register.

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_
        page 19, section B

    :param int num_bits_p: The number of bits to represent each dimension of the momentum register.
    :param bool prepare_catalytic_state: Flag for preparing the catalytic state from |0>. Set to True if the catalytic state has not already been prepared.
    """

    def __init__(self,num_bits_p:int,prepare_catalytic_state:bool=False):

        self.num_bits_p = num_bits_p
        self.prepare_catalytic_state = prepare_catalytic_state

    @cached_property
    def signature(self) -> Signature:
        return Signature([
            Register('w', BoundedQUInt(bitsize=2,iteration_length=3)),
            Register("r", BoundedQUInt(bitsize=self.num_bits_p)),
            Register("s", BoundedQUInt(bitsize=self.num_bits_p)),
            Register('less_than_ancilla', QBit()),
            Register('rot_ancilla', QBit()),
            Register('catalytic', QBit()),
            Register('flag_w', QBit(),side=Side.RIGHT)
            ])

    def pretty_name(self) -> str:
        return r'PREP T'

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        w_reg = quregs['w']
        r_reg = quregs['r']
        s_reg = quregs['s']
        less_than_ancilla = quregs['less_than_ancilla']
        rot_ancilla = quregs['rot_ancilla']
        catalytic = quregs['catalytic']
        flag_w = quregs['flag_w']

        yield FlaggedPrepareUniformSuperposition(d=3).on_registers(target=w_reg,less_than_ancilla=less_than_ancilla,rot_ancilla=rot_ancilla,success=flag_w)

        yield PreparePowerTwoState(bitsize=self.num_bits_p,prepare_catalytic_state=self.prepare_catalytic_state).on_registers(r=r_reg[:-1],success=r_reg[-1],catalytic=catalytic)

        yield PreparePowerTwoState(bitsize=self.num_bits_p).on_registers(r=s_reg[:-1],success=s_reg[-1],catalytic=catalytic)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(FlaggedPrepareUniformSuperposition(d=3),1), (PreparePowerTwoState(bitsize=self.num_bits_p,prepare_catalytic_state=self.prepare_catalytic_state), 1),(PreparePowerTwoState(bitsize=self.num_bits_p), 1)}