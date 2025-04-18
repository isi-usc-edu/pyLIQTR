"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
from functools import cached_property
from typing import Set

from qualtran import QInt, Signature, Register,Side,QBit,QAny
from qualtran.bloqs.chemistry.pbc.first_quantization.select_uv import SelectUVFirstQuantization
from qualtran.bloqs.arithmetic import Add
from qualtran.bloqs.basic_gates import Toffoli

from pyLIQTR.circuits.operators.SignedIntegerToTwosComplement import SignedIntegerToTwosComplement
from pyLIQTR.circuits.operators.ApplyNuclearPhase import ApplyNuclearPhase

class SelectUVFirstQuantizationPYL(SelectUVFirstQuantization):
    """SELECT for the U and V operators for the first quantized chemistry Hamiltonian.

    This does not include the controlled swaps from p_i and q_j system registers
    into ancilla registers and back again. Hence there is no system register.

    Registers:

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_

    :param int num_bits_p: The number of bits to represent each dimension of the momentum register.
    :param int eta: The number of electrons.
    :param int num_atoms: The number of atoms.
    :param int num_bits_nuc_pos: The number of bits to store each component of the
        nuclear positions. :math:`n_R` in the reference.
    """
    @cached_property
    def signature(self) -> Signature:
        n_nu = self.num_bits_p + 1
        return Signature(
            [
                Register("flag_tuv", QBit()),
                Register("uv", QBit(),side=Side.RIGHT),
                Register("l", QAny(bitsize=(self.num_atoms - 1).bit_length())),
                Register("rl", QAny(bitsize=self.num_bits_nuc_pos)),
                Register("nu", QAny(bitsize=n_nu), shape=(3,)),
                Register("p", QAny(bitsize=self.num_bits_p), shape=(3,)),
                Register("q", QAny(bitsize=self.num_bits_p), shape=(3,)),
            ]
        )

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        cost_tc = (SignedIntegerToTwosComplement(self.num_bits_p), 6)
        cost_add = (Add(QInt(self.num_bits_p + 1)), 6)  # + 2?
        cost_ctrl_add = (Toffoli(), 6 * (self.num_bits_p + 1))
        # + 2 as these numbers are larger from addition of $\nu$
        cost_inv_tc = (SignedIntegerToTwosComplement(self.num_bits_p + 2), 6)
        # 2. Phase by $e^{ik\cdot R}$ in the case of $U$ only.
        cost_phase = (ApplyNuclearPhase(self.num_bits_p, self.num_bits_nuc_pos), 1)
        return {cost_tc, cost_add, cost_ctrl_add, cost_inv_tc, cost_phase}