"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
import attrs
import qualtran as qt
from qualtran import GateWithRegisters, Signature
import numpy as np
from functools import cached_property
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
