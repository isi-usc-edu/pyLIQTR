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
import numpy as np
from qualtran import GateWithRegisters, Signature
from cirq._compat import cached_property
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