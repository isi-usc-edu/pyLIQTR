"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
from functools import cached_property
from typing import Dict, Set

from attrs import frozen

from qualtran import (
    Bloq,
    QAny,
    QBit,
    Register,
    Signature,
    SoquetT,
    Side
)
from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_zeta import PrepareZetaState

from pyLIQTR.circuits.operators.PrepareNu import PrepareNuState

@frozen
class PrepareUVFirstQuantization(Bloq):
    """PREPARE the U and V parts of the Hamiltonian.

    Registers:
        mu: The state controlling the nested boxes procedure.
        nu: The momentum transfer register.
        m: an ancilla register in a uniform superposition.
        l: The register for atomic species.
        flag_nu: Flag for success of the state preparation.
        catalytic: The catalytic state for controlled Hadamards.

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_
        page 21, section C

    :param int num_bits_p: The number of bits to represent each dimension of the momentum register.
    :param int eta: The number of electrons.
    :param int num_atoms: The number of atoms. :math:`L` in the reference.
    :param int m_param: :math:`$\\mathcal{M}$` in the reference.
    :param int lambda_zeta: sum of nuclear charges.
    :param int num_bits_nuc_pos: The number of bits of precision for representing the nuclear coordinates.
    """
    num_bits_p: int
    eta: int
    num_atoms: int
    m_param: int
    lambda_zeta: int
    num_bits_nuc_pos: int

    @cached_property
    def signature(self) -> Signature:
        n_m = (self.m_param - 1).bit_length()
        n_atom = (self.num_atoms - 1).bit_length()
        return Signature(
            [
                Register("mu", QAny(bitsize=self.num_bits_p)),
                Register("nu", QAny(bitsize=self.num_bits_p + 1), shape=(3,)),
                Register("m", QAny(bitsize=n_m)),
                Register("l", QAny(bitsize=n_atom)),
                Register("flag_nu", QBit()),
                Register("catalytic",QBit())
            ]
        )

    def pretty_name(self) -> str:
        return r'PREP UV'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {
            (PrepareNuState(self.num_bits_p, self.m_param), 1),
            (PrepareZetaState(self.num_atoms, self.lambda_zeta, self.num_bits_nuc_pos), 1),
        }