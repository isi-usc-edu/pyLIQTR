"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
from functools import cached_property
from typing import Set, Tuple
from numpy.typing import NDArray
from attrs import evolve, frozen

from qualtran import GateWithRegisters, QAny, QBit, Register, Side, Signature
from qualtran.bloqs.arithmetic.comparison import LinearDepthGreaterThan
from qualtran.bloqs.mcmt import MultiAnd

from pyLIQTR.circuits.operators.Product import Product
from pyLIQTR.circuits.operators.SumOfSquares import SumOf3Squares

@frozen
class NuInequalityTest(GateWithRegisters):
    """Tests the inequality

    .. math::
        (2^{\\mu-2})^2 \\mathcal{M} > m (\\nu_x^2 + \\nu_y^2 + \\nu_z^2)

    where :math:`m \\in [0, \\mathcal{M}-1]` and is stored in an ancilla register. This inequality is tested by computing :math:`(2^{\\mu-2})^2 \\mathcal{M}` and :math:`m (\\nu_x^2 + \\nu_y^2 + \\nu_z^2)` separately in two ancilla registers, which involves sum over squares and multiplication subcomponents.

    Registers:

    .. line-block::
        mu: the one-hot unary superposition register.
        nu: the momentum transfer register.
        m: the ancilla register in uniform superposition.
        flag_ineq: A flag from checking :math:`(2^{\\mu-2})^2 \\mathcal{M} > m (\\nu_x^2 + \\nu_y^2 + \\nu_z^2)`
        sos_product_ancilla: Ancilla used for bit products when summing over squares.
        sos_carry_ancilla: Ancilla used for carry bits when summing over squares.
        nu_mag_squared: Ancilla used to store :math:`\\nu_x^2 + \\nu_y^2 + \\nu_z^2`.
        m_times_nu: Ancilla used to store :math:`m (\\nu_x^2 + \\nu_y^2 + \\nu_z^2)`.


    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_
        Eq 89, pg 22-23

    :param int num_bits_p: The number of bits to represent each dimension of the nu register.
    :param int num_bits_m: The number of bits for :math:`\\mathcal{M}`. Eq 86.
    """
    num_bits_p: int
    num_bits_m: int

    @cached_property
    def ancilla_registers(self) -> Tuple[Register]:
        return (Register("sos_product_ancilla", QAny(bitsize=int(3*self.num_bits_p*(self.num_bits_p-1)/2)),side=Side.RIGHT),
        Register("sos_carry_ancilla",QAny(bitsize=int(self.num_bits_p*(3*self.num_bits_p+1)/2-3)),side=Side.RIGHT),
        Register("nu_mag_squared", QAny(bitsize=2*self.num_bits_p+2),side=Side.RIGHT),
        Register("m_times_nu", QAny(self.num_bits_m+2*self.num_bits_p+2), side=Side.RIGHT),
        )

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("mu", QAny(self.num_bits_p)),
                Register("nu", QAny(self.num_bits_p + 1), shape=(3,)),
                Register("m", QAny(self.num_bits_m)),
                Register("flag_ineq", QBit(), side=Side.RIGHT),
                *self.ancilla_registers
            ]
        )

    def pretty_name(self) -> str:
        return r'(2^(μ-2))^2 M > m ν^2'

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:
        
        mu, nu = quregs['mu'], quregs['nu']
        m = quregs['m']
        flag_ineq = quregs['flag_ineq']

        nu_mag_squared = quregs['nu_mag_squared'] # 2*np+2
        product_ancilla = quregs['sos_product_ancilla']
        carry_ancilla = quregs['sos_carry_ancilla']
        product_out = quregs['m_times_nu']

        nu_mags = nu[:,1:] # ignore sign bits

        # Compute $\nu_x^2 + \nu_y^2 + \nu_z^2$
        yield SumOf3Squares(self.num_bits_p).on_registers(input_vector=nu_mags,output=nu_mag_squared,product_ancilla=product_ancilla,carry_ancilla=carry_ancilla)

        # Compute $m (\nu_x^2 + \nu_y^2 + \nu_z^2)$
        ## B bitsize < A
        if self.num_bits_m < 2*self.num_bits_p+2:
            yield Product(2 * self.num_bits_p + 2, self.num_bits_m,).on_registers(A=nu_mag_squared,B=m,target=product_out)
        else:
            yield Product(self.num_bits_m, 2 * self.num_bits_p + 2).on_registers(A=m,B=nu_mag_squared,target=product_out)

        # "Compute" (2^{\mu-2})^2 * M
        ## first paragraph pg 23
        zeros_ancilla = context.qubit_manager.qalloc(self.num_bits_p+self.num_bits_m+2)
        two_mu_M=zeros_ancilla[self.num_bits_p+self.num_bits_m-5:]
        for i in range(self.num_bits_p):
            two_mu_M += [mu[i], zeros_ancilla[i]]
        two_mu_M += zeros_ancilla[self.num_bits_p:self.num_bits_p+self.num_bits_m-5]
        
        # Inequality test
        yield LinearDepthGreaterThan(bitsize=self.num_bits_m + 2 * self.num_bits_p + 2,signed=False).on_registers(a=two_mu_M,b=product_out,target=flag_ineq)

        context.qubit_manager.qfree(zeros_ancilla)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # 1. Compute $\nu_x^2 + \nu_y^2 + \nu_z^2$
        cost_1 = (SumOf3Squares(self.num_bits_p), 1)
        # 2. Compute $m (\nu_x^2 + \nu_y^2 + \nu_z^2)$
        if self.num_bits_m < 2*self.num_bits_p+2:
            cost_2 = (Product(2 * self.num_bits_p + 2, self.num_bits_m,), 1)
        else:
            cost_2 = (Product(self.num_bits_m, 2 * self.num_bits_p + 2), 1)
        # 3. Inequality test
        cost_3 = (LinearDepthGreaterThan(bitsize=self.num_bits_m + 2 * self.num_bits_p + 2,signed=False), 1)
        return {cost_1, cost_2, cost_3}