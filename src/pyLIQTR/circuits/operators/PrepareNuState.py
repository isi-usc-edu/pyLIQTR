"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
from functools import cached_property
from numpy.typing import NDArray
from typing import Set, Tuple

from qualtran import GateWithRegisters, QAny, QBit, Register, Side, Signature
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import PrepareUniformSuperposition
from qualtran.bloqs.mcmt import MultiAnd
from qualtran.symbolics.math_funcs import bit_length

from pyLIQTR.circuits.operators.PrepareMuUnaryEncoded import PrepareMuUnaryEncoded
from pyLIQTR.circuits.operators.PrepareNuSuperposition import PrepareNuSuperposition
from pyLIQTR.circuits.operators.FlagZeroAsFailure import FlagZeroAsFailure
from pyLIQTR.circuits.operators.FlagNuLessThanMu import FlagNuLessThanMu
from pyLIQTR.circuits.operators.NuInequalityTest import NuInequalityTest


class PrepareNuState(GateWithRegisters):
    """Prepares a state of the form

    .. math::
        \\frac{1}{\\sqrt{\\mathcal{M}2^{n_p + 2}}}
        \\sum_{\\mu=2}^{n_p+1}\\sum_{\\nu \\in B_\\mu}
        \\sum_{m=0}^{\\lceil \\mathcal M(2^{\mu-2}/\\lVert\\nu\\rVert)^2\\rceil-1}
        \\frac{1}{2^\mu}|\\mu\\rangle|\\nu_x\\rangle|\\nu_y\\rangle|\\nu_z\\rangle|m\\rangle|0\\rangle    

    This is a subcomponent of the PREPARE for the :math:`U` and :math:`V` components of the Hamiltonian in first quantization. 

    Registers:

    .. line-block::
        mu: The state controlling the nested boxes procedure.
        nu: The momentum transfer register.
        m: an ancilla register in a uniform superposition.
        flag_dim: Qubits flagging occurence of minus zero for each dimension of :math:`\\nu`.
        flag_minus_zero: Qubit flagging occurence of minus zero overall.
        flag_ineq: A flag from checking :math:`(2^{\\mu-2})^2 \\mathcal{M} > m (\\nu_x^2 + \\nu_y^2 + \\nu_z^2)`
        flag_nu_lt_mu: A flag from checking :math:`\\nu \\lt 2^{\\mu -2}`.
        flag_nu: Flag for success of the state preparation.
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

    :param int num_bits_p: The number of bits to represent each dimension of the momentum register :math:`|\\nu\\rangle`.
    :param int m_param: :math:`\\mathcal{M}` in the reference. Should be a power of two chosen to be sufficiently large to obtain an accurate block-encoding.
    """
    def __init__(self,num_bits_p:int,m_param:int):
        self.num_bits_p = num_bits_p
        self.m_param = m_param
        self.num_bits_m = bit_length(self.m_param - 1)

    @cached_property
    def ancilla_registers(self) -> Tuple[Register]:
        return (Register("flag_zero_and_ancilla", QAny(self.num_bits_p - 1), shape=(3,), side=Side.RIGHT),
        Register("flag_ancilla", QBit(), side=Side.RIGHT),
        Register("and_ancilla", QBit(),shape=(2,), side=Side.RIGHT),
        Register("nu_lt_mu_and_ancilla",QBit(),shape=(self.num_bits_p,2), side=Side.RIGHT),
        Register("sos_product_ancilla", QAny(bitsize=int(3*self.num_bits_p*(self.num_bits_p-1)/2)),side=Side.RIGHT),
        Register("sos_carry_ancilla",QAny(bitsize=int(self.num_bits_p*(3*self.num_bits_p+1)/2-3)),side=Side.RIGHT),
        Register("nu_mag_squared", QAny(bitsize=2*self.num_bits_p+2),side=Side.RIGHT),
        Register("m_times_nu", QAny(self.num_bits_m+2*self.num_bits_p+2), side=Side.RIGHT))

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("mu", QAny(bitsize=self.num_bits_p)),
                Register("nu", QAny(bitsize=self.num_bits_p + 1), shape=(3,)),
                Register("m", QAny(bitsize=self.num_bits_m)),
                Register("flag_dim", QBit(), shape=(3,), side=Side.RIGHT),
                Register("flag_minus_zero", QBit(), side=Side.RIGHT),
                Register("flag_ineq", QBit()),
                Register("flag_nu_lt_mu", QBit()),
                Register("flag_nu", QBit()),
                Register("catalytic",QBit()),
                *self.ancilla_registers
            ]
        )

    def pretty_name(self) -> str:
        return r"PREP 1/‖ν‖ ∣ν⟩"

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        mu, nu, m = quregs['mu'], quregs['nu'], quregs['m']
        catalytic = quregs['catalytic']
        flag_dim, flag_minus_zero, flag_nu_lt_mu, flag_ineq, flag_nu = quregs['flag_dim'], quregs['flag_minus_zero'], quregs['flag_nu_lt_mu'], quregs['flag_ineq'], quregs['flag_nu']
        flag_zero_and_ancilla, flag_ancilla = quregs['flag_zero_and_ancilla'], quregs['flag_ancilla']
        nu_lt_mu_and_ancilla, and_ancilla = quregs['nu_lt_mu_and_ancilla'], quregs['and_ancilla']
        sos_product_ancilla, sos_carry_ancilla, nu_mag_squared, m_times_nu = quregs['sos_product_ancilla'], quregs['sos_carry_ancilla'], quregs['nu_mag_squared'], quregs['m_times_nu']

        yield PrepareMuUnaryEncoded(self.num_bits_p).on_registers(mu=mu, catalytic=catalytic)

        yield PrepareNuSuperposition(self.num_bits_p).on_registers(mu=mu, nu=nu, catalytic=catalytic)

        yield FlagZeroAsFailure(self.num_bits_p).on_registers(nu=nu,flag_dim=flag_dim, flag_minus_zero=flag_minus_zero,and_ancilla=flag_zero_and_ancilla,flag_ancilla=flag_ancilla)

        yield FlagNuLessThanMu(self.num_bits_p).on_registers(mu=mu, nu=nu, flag_nu_lt_mu=flag_nu_lt_mu, and_ancilla=nu_lt_mu_and_ancilla)

        yield PrepareUniformSuperposition(self.m_param).on_registers(target=m)

        yield NuInequalityTest(self.num_bits_p, self.num_bits_m).on_registers(mu=mu, nu=nu, m=m,flag_ineq=flag_ineq, sos_product_ancilla=sos_product_ancilla, sos_carry_ancilla=sos_carry_ancilla, nu_mag_squared=nu_mag_squared, m_times_nu=m_times_nu)

        # flag overall success with 3 Ands
        ## four controls should be:
        ## 1 - failure of mu prep
        ## 2 - negative zero
        ## 3 - inner box (fbox)
        ## 4 - inequality test
        yield MultiAnd(cvs=(1,1,1,1)).on_registers(ctrl=[[mu[0]],flag_minus_zero, flag_nu_lt_mu, flag_ineq],junk=and_ancilla,target=flag_nu)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # 1. Prepare unary encoded superposition state (Eq 77)
        cost_1 = (PrepareMuUnaryEncoded(self.num_bits_p), 1)
        # 2. Prepare mu-nu superposition (Eq 78)
        cost_2 = (PrepareNuSuperposition(self.num_bits_p), 1)
        # 3. Remove minus zero
        cost_3 = (FlagZeroAsFailure(self.num_bits_p), 1)
        # 4. Test $\nu < 2^{\mu-2}$
        cost_4 = (FlagNuLessThanMu(self.num_bits_p), 1)
        # 5. Prepare superposition over $m$ which is a power of two so only clifford.
        cost_5 = (PrepareUniformSuperposition(self.m_param),1)
        # 6. Test that $(2^{\mu-2})^2\mathcal{M} > m (\nu_x^2 + \nu_y^2 + \nu_z^2)$
        cost_6 = (NuInequalityTest(self.num_bits_p, self.num_bits_m), 1)
        # 7. Flag overall success
        cost_7 = (MultiAnd(cvs=(1,1,1,1)),1)
        return {cost_1, cost_2, cost_3, cost_4, cost_5, cost_6, cost_7}