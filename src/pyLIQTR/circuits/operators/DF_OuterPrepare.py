"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
findings, conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the Under Secretary of Defense
for Research and Engineering.

© 2023 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
above. Use of this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.
"""
from numpy.typing import NDArray
from typing import Tuple

import cirq
import qualtran as qt
from qualtran.bloqs.state_preparation import StatePreparationAliasSampling
from qualtran.bloqs.arithmetic import LessThanEqual
from qualtran.bloqs.basic_gates.swap import CSwap
from qualtran.bloqs.prepare_uniform_superposition import PrepareUniformSuperposition
from qualtran.bloqs.qrom import QROM
from cirq._compat import cached_property
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
        success = quregs['success']
        N = self.selection_registers[0].iteration_length

        yield FlaggedPrepareUniformSuperposition(N).on_registers(target=selection,success=success,less_than_ancilla=less_than_equal,rot_ancilla=alt[0])

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