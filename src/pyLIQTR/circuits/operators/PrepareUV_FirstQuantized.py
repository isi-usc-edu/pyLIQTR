"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
from functools import cached_property
from typing import Set, Tuple
from numpy.typing import NDArray

from attrs import frozen

from qualtran import (
    GateWithRegisters,
    QAny,
    QBit,
    Register,
    Signature,
    SoquetT,
    Side
)
from qualtran.symbolics.math_funcs import bit_length

from pyLIQTR.circuits.operators.PrepareNuState import PrepareNuState
from pyLIQTR.circuits.operators.PrepareZetaState import PrepareZetaState

@frozen
class PrepareUV_FirstQuantized(GateWithRegisters):
    """PREPARE the U and V parts of the Hamiltonian in first quantization.

    Registers:

    .. line-block::
        mu: The state controlling the nested boxes procedure.
        nu: The momentum transfer register.
        m: an ancilla register in a uniform superposition.
        superposition_state: Register in a superposition with relative amplitudes :math:`\\sqrt{\\lambda_U}` and :math:`\\sqrt{\\lambda_V}` used to select between U and V.
        Rl: Register to store the value of the nuclear coordinates :math:`R_\\ell`.
        flag_dim: Qubits flagging occurence of minus zero for each dimension of :math:`\\nu`.
        flag_minus_zero: Qubit flagging occurence of minus zero overall.
        flag_ineq:  A flag from checking :math:`(2^{\\mu-2})^2 \\mathcal{M} > m (\\nu_x^2 + \\nu_y^2 + \\nu_z^2)`.
        flag_nu_lt_mu: A flag from checking :math:`\\nu \\lt 2^{\\mu -2}`.
        flag_nu: Flag for success of the :math:`\\frac{1}{|\\nu|}` state preparation.
        flag_ineq_uv: Flag from the inequality test with :math:`\\eta` controlling the relative amplitude between U and V.
        catalytic: The catalytic state for controlled Hadamards.
        flag_zero_and_ancilla: Ancilla used for the And ladder when flagging minus zero.
        flag_ancilla: Ancilla used for OR to mark flag_minus_zero.
        and_ancilla: Ancilla used for flagging overall success.
        nu_lt_mu_and_ancilla: Ancilla used for the And ladder when checking :math:`\\nu \\lt 2^{\\mu -2}.
        sos_product_ancilla: Ancilla used for bit products when summing over squares.
        sos_carry_ancilla: Ancilla used for carry bits when summing over squares.
        nu_mag_squared: Ancilla used to store :math:`\\nu_x^2 + \\nu_y^2 + \\nu_z^2`.
        m_times_nu: Ancilla used to store :math:`m (\\nu_x^2 + \\nu_y^2 + \\nu_z^2)`.

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_
        page 21, section C

    :param int num_bits_p: The number of bits to represent each dimension of the momentum register.
    :param int eta: The number of electrons.
    :param int num_atoms: The number of atoms. :math:`L` in the reference.
    :param int m_param: :math:`$\\mathcal{M}$` in the reference.
    :param int lambda_zeta: sum of nuclear charges.
    :param int num_bits_nuc_pos: The number of bits of precision for representing the nuclear coordinates :math:`R_\ell`.
    """
    num_bits_p: int
    eta: int
    num_atoms: int
    m_param: int
    lambda_zeta: int
    num_bits_nuc_pos: int

    @cached_property
    def ancilla_registers(self) -> Tuple[Register]:
        n_m = (self.m_param - 1).bit_length()
        return (Register("flag_zero_and_ancilla", QAny(self.num_bits_p - 1), shape=(3,), side=Side.RIGHT),
        Register("flag_ancilla", QBit(), side=Side.RIGHT),
        Register("and_ancilla", QBit(),shape=(2,), side=Side.RIGHT),
        Register("nu_lt_mu_and_ancilla",QBit(),shape=(self.num_bits_p,2), side=Side.RIGHT),
        Register("sos_product_ancilla", QAny(bitsize=int(3*self.num_bits_p*(self.num_bits_p-1)/2)),side=Side.RIGHT),
        Register("sos_carry_ancilla",QAny(bitsize=int(self.num_bits_p*(3*self.num_bits_p+1)/2-3)),side=Side.RIGHT),
        Register("nu_mag_squared", QAny(bitsize=2*self.num_bits_p+2),side=Side.RIGHT),
        Register("m_times_nu", QAny(n_m+2*self.num_bits_p+2), side=Side.RIGHT))

    @cached_property
    def signature(self) -> Signature:
        n_m = (self.m_param - 1).bit_length()
        n_eta_zeta = bit_length(self.eta + 2 * self.lambda_zeta - 1)
        return Signature(
            [
                Register("mu", QAny(bitsize=self.num_bits_p)),
                Register("nu", QAny(bitsize=self.num_bits_p + 1), shape=(3,)),
                Register("m", QAny(bitsize=n_m)),
                Register("superposition_state", QAny(bitsize=n_eta_zeta)),
                Register("Rl", QAny(bitsize=self.num_bits_nuc_pos), shape=(3,)),
                Register("flag_dim", QBit(), shape=(3,), side=Side.RIGHT),
                Register("flag_minus_zero", QBit(), side=Side.RIGHT),
                Register("flag_ineq", QBit()),
                Register("flag_nu_lt_mu", QBit()),
                Register("flag_nu", QBit()),
                Register("flag_ineq_uv", QBit()),
                Register("catalytic",QBit()),
                *self.ancilla_registers
            ]
        )

    def pretty_name(self) -> str:
        return r'PREP UV'

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:
        flag_inequality = quregs['flag_ineq_uv']
        zeta_select = quregs['superposition_state']
        Rl = quregs['Rl']

        yield PrepareNuState(self.num_bits_p, self.m_param).on_registers(**quregs)

        yield PrepareZetaState(self.num_atoms, self.lambda_zeta, self.num_bits_nuc_pos,self.eta).on_registers(control=flag_inequality,select=zeta_select,Rl=Rl)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {
            (PrepareNuState(self.num_bits_p, self.m_param), 1),
            (PrepareZetaState(self.num_atoms, self.lambda_zeta, self.num_bits_nuc_pos,self.eta), 1),
        }