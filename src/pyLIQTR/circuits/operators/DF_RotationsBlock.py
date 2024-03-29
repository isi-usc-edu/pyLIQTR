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
import cirq
import attrs
import qualtran as qt
from qualtran import GateWithRegisters, Signature
import numpy as np
from cirq._compat import cached_property
from typing import Sequence, Tuple
from numpy.typing import NDArray
from pyLIQTR.circuits.operators.ProductPauliExponentials import ProductPauliExponentials

@attrs.frozen
class RotationsBlock(GateWithRegisters):

    num_data_bits: int
    num_target_bits: int
    precision_bits: int
    phase_gradient_bits: int
    uncompute: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(
            angle_data=self.num_data_bits,
            target=self.num_target_bits,
            phi=self.phase_gradient_bits #phase gradient state
        )

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:

        angle_data = quregs['angle_data']
        target = quregs['target']
        phase_gradient_state = quregs['phi']

        exp_product_gate = ProductPauliExponentials(self.precision_bits,bphi=self.phase_gradient_bits,uncompute=self.uncompute)

        for i in range(self.num_target_bits-1):
            yield exp_product_gate.on_registers(angle_data=angle_data[i*self.precision_bits:(i+1)*self.precision_bits],target=target[i:i+2],phi=phase_gradient_state)
