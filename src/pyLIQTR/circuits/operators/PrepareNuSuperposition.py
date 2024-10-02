"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
from functools import cached_property
from numpy.typing import NDArray
from typing import Set
from qualtran import GateWithRegisters, Signature, QAny, QBit, Register
from qualtran.bloqs.basic_gates import Hadamard, TGate
from pyLIQTR.circuits.operators.CatalyticallyControlledHadamard import CatalyticallyControlledHadamard

class PrepareNuSuperposition(GateWithRegisters):
    """Prepares the superposition state

    .. math::
        \\frac{1}{ \\sqrt{2^{n_p + 2}} } \\sum_{\\mu=2}^{n_p+1} \\sum_{\\nu_x,\\nu_y,\\nu_z=(-2^{\\mu-1}-1),-0}^{2^{\\mu-1}-1} \frac{1}{2^{\\mu}} |\\mu\\rangle |\\nu_x\\rangle |\\nu_y\\rangle |\\nu_z\\rangle ,

    where the :math:`-0` indicates the sum includes the state representing negative zero since the signed integer representation is being used. This is done using 6 Hadamards and 3*(bitsize+1) catalytically controlled Hadamards. This assumes :math:`|\\mu\\rangle` is given in the state 

    .. math::
        \\frac{1}{ \\sqrt{2^{n_p + 2}} } \\sum_{\\mu=2}^{n_p+1} 2^{\\mu/2} |0\\cdots 0\\underbrace{1\\cdots 1}_{\\mu-1}\\rangle + \\frac{1}{ \\sqrt{2^{n_p}} }|0\\cdots 0\\rangle .

    which can be prepared using :class:`PrepareMuUnaryEncoded`.

    Registers:
        mu: The register used to control the Hadamards.
        nu: The target register for the state preparation.
        catalytic: The catalytic state for controlled Hadamards.

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_
        Eq 78-79

    :param int bitsize: the number of bits :math:`n_p`.
    :param bool prepare_catalytic_state: Flag for preparing the catalytic state from |0>. Set to True if the catalytic state has not already been prepared.
    """

    def __init__(self,bitsize:int,prepare_catalytic_state:bool=False):
        self.bitsize = bitsize 
        self.prepare_catalytic_state = prepare_catalytic_state

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("mu", QAny(self.bitsize)),
                Register("nu", QAny(self.bitsize + 1), shape=(3,)), #+1 for sign qubit
                Register("catalytic", QBit())
            ]
        )

    def pretty_name(self) -> str:
        return r'PREP (2^-μ)|μ⟩|ν⟩'

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        mu_reg = quregs['mu']
        nu_reg = quregs['nu']
        catalytic = quregs['catalytic']

        for i,dim_qbs in enumerate(nu_reg):

            # these are uncontrolled since min value of mu is 2
            yield Hadamard().on(dim_qbs[0]) # sign qubit
            yield Hadamard().on(dim_qbs[-1])

            for bit in range(1,self.bitsize):
                if i==0 and bit == 1:
                # need to prepare the catalytic state on the first call
                    yield CatalyticallyControlledHadamard(prepare_catalytic_state=self.prepare_catalytic_state).on_registers(control=mu_reg[self.bitsize-bit],target=dim_qbs[self.bitsize-bit],catalytic=catalytic)
                else:
                    yield CatalyticallyControlledHadamard(prepare_catalytic_state=False).on_registers(control=mu_reg[self.bitsize-bit],target=dim_qbs[self.bitsize-bit],catalytic=catalytic)


    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        if self.prepare_catalytic_state:
            return {(CatalyticallyControlledHadamard(), 3*(self.bitsize-1)),(Hadamard(),7),(TGate(),1)}
        return {(CatalyticallyControlledHadamard(), 3*(self.bitsize-1)),(Hadamard(),6)}
