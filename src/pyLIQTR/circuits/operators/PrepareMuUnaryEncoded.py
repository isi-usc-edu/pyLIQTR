"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
from functools import cached_property
from numpy.typing import NDArray
from typing import Dict, Set
from qualtran import GateWithRegisters, Signature
from qualtran.bloqs.basic_gates import Hadamard, XGate, TGate
from qualtran.bloqs.basic_gates.on_each import OnEach
import cirq
from pyLIQTR.circuits.operators.CatalyticallyControlledHadamard import CatalyticallyControlledHadamard

class PrepareMuUnaryEncoded(GateWithRegisters):
    """Prepares the unary-encoded superposition state

    .. math::
        \\frac{1}{ \\sqrt{2^{n_p + 2}} } \\sum_{\\mu=2}^{n_p+1} 2^{\\mu/2} |0\\cdots 0\\underbrace{1\\cdots 1}_{\\mu-1}\\rangle + \\frac{1}{ \\sqrt{2^{n_p}} }|0\\cdots 0\\rangle .

    Registers:
        mu: The register we want to prepare the state over.
        catalytic: The catalytic state for controlled Hadamards.

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_
        Eq 77

    :param int bitsize: the number of bits :math:`n_p`.
    :param bool prepare_catalytic_state: Flag for preparing the catalytic state from |0>. Set to True if the catalytic state has not already been prepared.
    """

    def __init__(self,bitsize:int,prepare_catalytic_state:bool=False):
        self.bitsize = bitsize 
        self.prepare_catalytic_state = prepare_catalytic_state

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(mu=self.bitsize,catalytic=1)

    def pretty_name(self) -> str:
        return r'PREP √(2^μ)|μ⟩'

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        mu_reg = quregs['mu']
        catalytic = quregs['catalytic']

        # first, prepare the state 2^((-bitsize-2)/2)*2^(mu/2)|1...10...0> in unary using one Hadamard followed by bitsize-1 controlled Hadamards
        yield Hadamard().on(mu_reg[0])
        for bit in range(self.bitsize-1):
            if bit == 0:
                # need to prepare the catalytic state on the first call
                yield CatalyticallyControlledHadamard(prepare_catalytic_state=self.prepare_catalytic_state).on_registers(control=mu_reg[bit],target=mu_reg[bit+1],catalytic=catalytic)
            else:
                yield CatalyticallyControlledHadamard(prepare_catalytic_state=False).on_registers(control=mu_reg[bit],target=mu_reg[bit+1],catalytic=catalytic)
                
        # next, flip all the qubits to get the correct coefficients with the correct states, ie |1...10...0> -> |0...01...1>
        yield OnEach(self.bitsize, XGate()).on_registers(q=mu_reg)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        if self.prepare_catalytic_state:
            return {(CatalyticallyControlledHadamard(), self.bitsize-1),(Hadamard(),2),(XGate(),self.bitsize),(TGate(),1)}
        return {(CatalyticallyControlledHadamard(), self.bitsize-1),(Hadamard(),1),(XGate(),self.bitsize)}
