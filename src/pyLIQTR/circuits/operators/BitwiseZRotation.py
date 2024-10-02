"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
import attrs
import numpy as np
from qualtran import GateWithRegisters, Signature
from functools import cached_property
from numpy.typing import NDArray

@attrs.frozen
class BitwiseZRotation(GateWithRegisters):
    '''
    Implements 

    U|theta>|target> -> |theta>exp(i*2pi*theta*Z)|target>

    where |theta> is a br-qubit register storing the binary representation of the angle theta and |target> is the single qubit the rotation acts on.
    '''
    br: int # number of bits to represent angle
    theta_sign: float = 1.0 # set to -1.0 for exp(-i*2pi*theta*Z)

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(
            angle_data=self.br,
            target=1
        )

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:

        angle_reg = quregs['angle_data']
        target = quregs['target']

        for b in range(self.br):
            theta_b = self.theta_sign*2*np.pi/(2**(1+b))
            # cirq.Rz gate is exp(-iZ<rads>/2)
            yield cirq.Rz(rads=-2*theta_b).on(*target).controlled_by(angle_reg[b])