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
from typing import Tuple

import attrs
import cirq
import numpy as np
from cirq._compat import cached_property
from numpy.typing import NDArray

from qualtran import GateWithRegisters, Signature
from qualtran.bloqs.and_bloq import And, MultiAnd
from qualtran.bloqs.arithmetic import LessThanConstant
from qualtran.bloqs.multi_control_multi_target_pauli import MultiControlPauli

class FlaggedPrepareUniformSuperposition(GateWithRegisters):
    '''
    prepares uniform superposition over d basis states according to procedure outlined in https://arxiv.org/abs/2011.03494 Appendix A page 39 with the added step of flagging success. Toffoli cost nominally 3 logd - 3*\eta + 2*br - 9, where 2^\eta is a factor of d and br is the rotation gate precision bits
    '''
    def __init__(self,d:int):
        self.d = d # number of basis states
        self.nd_bits = (d-1).bit_length()

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(
            target=self.nd_bits,
            less_than_ancilla=1,
            rot_ancilla=1,
            success=1
        )

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        target = quregs['target']
        less_than,success_ancilla, rot_ancilla = quregs['less_than_ancilla'],quregs['success'], quregs['rot_ancilla']
        # find eta and logL = log(d)-eta
        d, eta = self.d, 0
        while d > 1 and d % 2 == 0:
            eta += 1
            d = d // 2
        l, logL = int(d), len(target) - eta
        logL_qubits = target[:logL]
        # l, logL = int(d), len(target)
        # logL_qubits = target

        yield cirq.H.on_each(*target)

        # iii. Inequality test between target register and l
        yield LessThanConstant(logL, l).on_registers(x=logL_qubits, target=less_than)

        # iv. Rotate ancilla qubit
        rotation_angle = np.arccos(1 - (2 ** np.floor(np.log2(l))) / l)
        yield cirq.Ry(rads=rotation_angle)(*rot_ancilla)

        # v. Reflection on result of inequality test and ancilla
        yield cirq.CZ(*less_than,*rot_ancilla)

        # vi. Invert rotation
        yield cirq.Ry(rads=-rotation_angle)(*rot_ancilla)
        ## and invert inequality test
        yield LessThanConstant(logL, l).on_registers(x=logL_qubits, target=less_than)

        # vii. Repeat Hadamards
        yield cirq.H.on_each(*target)

        # viii. Reflect about zero state 
        yield MultiControlPauli(cvs=(1,)*self.nd_bits).on_registers(controls=target,target=rot_ancilla)

        # ix. Repeat Hadamards. 
        yield cirq.H.on_each(*target)

        # x. Repeat inequality test
        yield LessThanConstant(logL, l).on_registers(x=logL_qubits, target=less_than)

        ## success flagged on inequality test
        yield cirq.CNOT(*less_than,*success_ancilla)

        # Invert inequality test
        yield LessThanConstant(logL, l).on_registers(x=logL_qubits, target=less_than)