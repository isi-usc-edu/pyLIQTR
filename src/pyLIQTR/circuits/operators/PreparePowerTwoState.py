"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
from functools import cached_property
from numpy.typing import NDArray
from typing import Dict, Set
from qualtran import GateWithRegisters, Signature
from qualtran.bloqs.basic_gates import TGate, Hadamard, XGate, CNOT
from qualtran.bloqs.basic_gates.on_each import OnEach
import cirq
from pyLIQTR.circuits.operators.CatalyticallyControlledHadamard import CatalyticallyControlledHadamard

class PreparePowerTwoState(GateWithRegisters):
    """Prepares the uniform superposition over |r> given by Eq. 69 in the reference.

    This prepares the state

    .. math::
        2^{(-n_p -1)/2} \\sum_r=0^{n_p-2} 2^{r/2} |r\\rangle|0\\rangle + 2^{(-n_p -1)/2}|0\\rangle|1\\rangle

    where |r> is in one-hot unary on :math:`n_p-1` qubits and the additional qubit flags success with 0. Note, the convention used here is that the state |100..> corresponds to r=0, |010..> to r=1, and so on.

    Parameters:
        bitsize: the number of bits :math:`n_p`.

    Registers:
        r: The register we want to prepare the state over.
        success: The qubit flagging success of the preparation with 0.
        catalytic: The catalytic state for controlled Hadamards.

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_
        Eq 67-69, pg 19-20
    """

    def __init__(self,bitsize:int,prepare_catalytic_state:bool=False):
        self.bitsize = bitsize 
        self.prepare_catalytic_state = prepare_catalytic_state

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(r=self.bitsize-1,success=1,catalytic=1)

    def pretty_name(self) -> str:
        return r'PREP 2^(r/2) |r⟩'

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        r_reg = quregs['r']
        success = quregs['success']
        catalytic = quregs['catalytic']

        # first, prepare the state 2^{-(r+1)/2}|r> in unary using one Hadamard followed by bitsize-1 controlled Hadamards
        yield Hadamard().on(r_reg[-1])
        for bit in range(self.bitsize-2):
            if bit == 0:
                # need to prepare the catalytic state on the first call
                yield CatalyticallyControlledHadamard(prepare_catalytic_state=self.prepare_catalytic_state).on_registers(control=r_reg[self.bitsize-(bit+2)],target=r_reg[self.bitsize-(bit+3)],catalytic=catalytic)
            else:
                yield CatalyticallyControlledHadamard(prepare_catalytic_state=False).on_registers(control=r_reg[self.bitsize-(bit+2)],target=r_reg[self.bitsize-(bit+3)],catalytic=catalytic)
                
        # next, flip all the qubits and prepare the flag qubit in the |1> state. Then the state becomes
        #\sum_r=0^{bitsize-2}2^{-(r+1)/2}|bitsize-2-r> = 2^{(-bitsize-1)/2} \sum_r=0^{bitsize-2} 2^{r/2}|r>
        # yield cirq.X.on_each([*r_reg,*success])
        yield OnEach(self.bitsize, XGate()).on_registers(q=[*r_reg,*success])

        # finally, convert to one-hot unary using a sequence of CNOTs. After this step, the success qubit flags success with |0>
        yield CNOT().on(r_reg[0],*success)
        for bit in range(self.bitsize-2):
            yield CNOT().on(r_reg[bit+1],r_reg[bit])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        if self.prepare_catalytic_state:
            return {(CatalyticallyControlledHadamard(), (self.bitsize - 2)),(Hadamard(),2),(XGate(),self.bitsize),(CNOT(),self.bitsize-1),(TGate(),1)}
        return {(CatalyticallyControlledHadamard(), (self.bitsize - 2)),(Hadamard(),1),(XGate(),self.bitsize),(CNOT(),self.bitsize-1)}
