"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
from functools import cached_property
from typing import Dict, Set, TYPE_CHECKING

from attrs import evolve, frozen

from qualtran import Bloq, BloqBuilder, QAny, QBit, Register, Side, Signature, SoquetT
from qualtran.bloqs.arithmetic import GreaterThan, Product, SumOfSquares
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
    PrepareUniformSuperposition,
)
from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu import TestNuLessThanMu

from pyLIQTR.circuits.operators.PrepareMuUnaryEncoded import PrepareMuUnaryEncoded
from pyLIQTR.circuits.operators.PrepareNuSuperposition import PrepareNuSuperposition
from pyLIQTR.circuits.operators.FlagZeroAsFailure import FlagZeroAsFailure

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class TestNuInequality(Bloq):
    """Bloq to flag if all components of :math:`$\\nu$` are smaller in absolute value than :math:`2^{\\mu-2}`.

    Tests the inequality

    .. math::
        (2^{\\mu-2})^2 \\mathcal{M} > m (\\nu_x^2 + \\nu_y^2 + \\nu_z^2)

     where $m \in [0, \mathcal{M}-1]$ and is stored in an ancilla register. This is done by computing :math:`(2^{\\mu-2})^2 \\mathcal{M}` and :math:`m (\\nu_x^2 + \\nu_y^2 + \\nu_z^2)` separately in two ancilla registers, which involves sum of squares and multiplication subcomponents.

    Registers:
        mu: the one-hot unary superposition register.
        nu: the momentum transfer register.
        m: the ancilla register in uniform superposition.
        flag_minus_zero: A flag for occurence of negative zero.
        flag_ineq: A flag from checking $\nu \lt 2^{\mu -2}$.
        succ: a flag bit for failure of the state nu preparation.

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_
        Eq 89, pg 22-23

    :param int num_bits_p: The number of bits to represent each dimension of the nu register.
    :param int num_bits_m: The number of bits for :math:`$\\mathcal{M}$`. Eq 86.
    :param bool is_adjoint: Whether to do the bloq adjoint or not.
    """
    num_bits_p: int
    num_bits_m: int
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("mu", QAny(self.num_bits_p)),
                Register("nu", QAny(self.num_bits_p + 1), shape=(3,)),
                Register("m", QAny(self.num_bits_m)),
                Register(
                    "flag_minus_zero", QBit(),
                ),
                Register(
                    "flag_ineq", QBit(), side=Side.LEFT if not self.is_adjoint else Side.RIGHT
                ),
                Register("succ", QBit()),
            ]
        )

    def pretty_name(self) -> str:
        return r'(2^(μ-2))^2 M > m ν^2'

    def adjoint(self) -> 'Bloq':
        return evolve(self, is_adjoint=not self.is_adjoint)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        if self.is_adjoint:
            return {(Toffoli(), 0)}
        else:
            # 1. Compute $\nu_x^2 + \nu_y^2 + \nu_z^2$
            cost_1 = (SumOfSquares(self.num_bits_p, k=3), 1)
            # 2. Compute $m (\nu_x^2 + \nu_y^2 + \nu_z^2)$
            cost_2 = (Product(2 * self.num_bits_p + 2, self.num_bits_m), 1)

            # 3. Inequality test
            cost_3 = (GreaterThan(self.num_bits_m, 2 * self.num_bits_p + 2), 1)
            # 4. 3 Toffoli for overall success
            cost_4 = (Toffoli(), 3)
            return {cost_1, cost_2, cost_3, cost_4}


@frozen
class PrepareNuState(Bloq):
    """Prepares a state of the form

    .. math::
        \\frac{1}{\\sqrt{\\mathcal{M}2^{n_p + 2}}}
        \\sum_{\\mu=2}^{n_p+1}\\sum_{\\nu \\in B_\\mu}
        \\sum_{m=0}^{\\lceil \\mathcal M(2^{\mu-2}/\\lVert\\nu\\rVert)^2\\rceil-1}
        \\frac{1}{2^\mu}|\\mu\\rangle|\\nu_x\\rangle|\\nu_y\\rangle|\\nu_z\\rangle|m\\rangle|0\\rangle    

    This is a subcomponent of the PREPARE for the :math:`U` and :math:`V` components of the Hamiltonian. 

    Registers:
        mu: The state controlling the nested boxes procedure.
        nu: The momentum transfer register.
        flag_dim: Qubits flagging occurence of minus zero for each dimension of :math:`\\nu`.
        flag_minus_zero: Qubit flagging occurence of minus zero overall.
        and_ancilla: Ancilla used for the And ladder.
        flag_ancilla: Ancilla used for OR to mark flag_minus_zero.
        m: an ancilla register in a uniform superposition.
        flag_nu: Flag for success of the state preparation.
        catalytic: The catalytic state for controlled Hadamards.

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_
        page 21, section C

    :param int num_bits_p: The number of bits to represent each dimension of the momentum register :math:`|\\nu\\rangle`.
    :param int m_param: :math:`$\\mathcal{M}$` in the reference. Should be a power of two chosen to be sufficiently large to obtain an accurate block-encoding.
    """
    num_bits_p: int
    m_param: int

    @cached_property
    def signature(self) -> Signature:
        n_m = (self.m_param - 1).bit_length()
        return Signature(
            [
                Register("mu", QAny(bitsize=self.num_bits_p)),
                Register("nu", QAny(bitsize=self.num_bits_p + 1), shape=(3,)),
                Register("flag_dim", QBit(), shape=(3,), side=Side.RIGHT),
                Register("flag_minus_zero", QBit(), side=Side.RIGHT),
                Register("and_ancilla", QAny(self.num_bits_p - 1), shape=(3,), side=Side.RIGHT),
                Register("flag_ancilla", QBit(), side=Side.RIGHT),
                Register("m", QAny(bitsize=n_m)),
                Register("flag_nu", QBit()),
                Register("catalytic",QBit())
            ]
        )

    def pretty_name(self) -> str:
        return r"PREP 1/‖ν‖ ∣ν⟩"

    def build_composite_bloq(
        self, bb: BloqBuilder, mu: SoquetT, nu: SoquetT, m: SoquetT, flag_nu: SoquetT, catalytic: SoquetT
    ) -> Dict[str, 'SoquetT']:
        mu, catalytic = bb.add(PrepareMuUnaryEncoded(self.num_bits_p), mu=mu, catalytic=catalytic)
        mu, nu, catalytic = bb.add(PrepareNuSuperposition(self.num_bits_p), mu=mu, nu=nu, catalytic=catalytic)
        nu, flag_dim, flag_minus_zero, and_ancilla, flag_ancilla = bb.add(FlagZeroAsFailure(self.num_bits_p), nu=nu)
        mu, nu, flag_nu_lt_mu = bb.add(TestNuLessThanMu(self.num_bits_p), mu=mu, nu=nu)
        n_m = (self.m_param - 1).bit_length()
        m = bb.add(PrepareUniformSuperposition(self.m_param), target=m)

        mu, nu, m, flag_minus_zero, flag_nu = bb.add(
            TestNuInequality(self.num_bits_p, n_m),
            mu=mu,
            nu=nu,
            m=m,
            flag_minus_zero=flag_minus_zero,
            flag_ineq=flag_nu_lt_mu,
            succ=flag_nu,
        )
        return {'mu': mu, 'nu': nu, 'm': m, 'flag_nu': flag_nu, 'catalytic':catalytic, 'flag_dim': flag_dim, 'flag_minus_zero': flag_minus_zero, 'and_ancilla': and_ancilla, 'flag_ancilla': flag_ancilla}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # 1. Prepare unary encoded superposition state (Eq 77)
        cost_1 = (PrepareMuUnaryEncoded(self.num_bits_p), 1)
        n_m = (self.m_param - 1).bit_length()
        # 2. Prepare mu-nu superposition (Eq 78)
        cost_2 = (PrepareNuSuperposition(self.num_bits_p), 1)
        # 3. Remove minus zero
        cost_3 = (FlagZeroAsFailure(self.num_bits_p), 1)
        # 4. Test $\nu < 2^{\mu-2}$
        cost_4 = (TestNuLessThanMu(self.num_bits_p), 1)
        # 5. Prepare superposition over $m$ which is a power of two so only clifford.
        cost_5 = (PrepareUniformSuperposition(self.m_param),1)
        # 6. Test that $(2^{\mu-2})^2\mathcal{M} > m (\nu_x^2 + \nu_y^2 + \nu_z^2)$
        cost_6 = (TestNuInequality(self.num_bits_p, n_m), 1)
        return {cost_1, cost_2, cost_3, cost_4, cost_5, cost_6}