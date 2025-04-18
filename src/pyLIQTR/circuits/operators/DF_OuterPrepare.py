"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
from numpy.typing import NDArray
from typing import Tuple

import cirq
import qualtran as qt
from qualtran.bloqs.state_preparation import StatePreparationAliasSampling
from qualtran.bloqs.arithmetic import LessThanEqual
from qualtran.bloqs.basic_gates.swap import CSwap
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import PrepareUniformSuperposition
from qualtran.bloqs.data_loading.qrom import QROM
from functools import cached_property
from qualtran._infra.registers import Register

from pyLIQTR.circuits.operators.FlaggedPrepareUniformSuperposition import FlaggedPrepareUniformSuperposition

class OuterPrepare(StatePreparationAliasSampling):
    '''
    Implements first block [prep] in Appendix C Fig 16 from https://arxiv.org/abs/2011.03494
    (Step 1 pg 51-52)
    '''
    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return tuple(
            Signature.build(
                sigma_mu=self.sigma_mu_bitsize,
                alt=self.alternates_bitsize,
                keep=self.keep_bitsize,
                rot_ancilla=1,
                less_than_equal=1,
                success = 1
            )
        )
     
    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        selection, less_than_equal = quregs['selection'], quregs['less_than_equal']
        sigma_mu, alt, keep = quregs.get('sigma_mu', ()), quregs['alt'], quregs.get('keep', ())
        success, rot_ancilla = quregs['success'], quregs['rot_ancilla']
        N = self.selection_registers[0].iteration_length

        yield FlaggedPrepareUniformSuperposition(N).on_registers(target=selection,success=success,less_than_ancilla=less_than_equal,rot_ancilla=rot_ancilla)

        yield cirq.H.on_each(*sigma_mu)

        qrom_gate = QROM(
            [self.alt, self.keep],
            (self.selection_bitsize,),
            (self.alternates_bitsize, self.keep_bitsize),
        )
        yield qrom_gate.on_registers(selection=selection, target0_=alt, target1_=keep)

        yield LessThanEqualGate(self.mu, self.mu).on(
            *keep, *sigma_mu, *less_than_equal
        )
        yield CSwap.make_on(
            ctrl=less_than_equal, x=alt, y=selection
        )

        # uncompute less than equal
        yield LessThanEqualGate(self.mu, self.mu).on(
            *keep, *sigma_mu, *less_than_equal
        )