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

import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from numpy.typing import NDArray

from qualtran import GateWithRegisters, Signature
from qualtran.bloqs.and_bloq import And, MultiAnd
from qualtran.bloqs.arithmetic import LessThanEqual
from qualtran.bloqs.multi_control_multi_target_pauli import MultiControlPauli

from pyLIQTR.circuits.operators.BinaryToUnary import BinaryToUnaryBits
from pyLIQTR.circuits.operators.PhaseGradientRotation import PhaseGradientZRotation


@attr.frozen
class ControlledPrepareUniformSuperposition(GateWithRegisters):
    '''
    Implements step 3a on page 52 of the reference.

    Prepares the |p> register in an equal superposition of 2**⎡log2(Xi_l)⎤ states, where states satisfying |state> <= |Xi_l> are flagged on an ancilla 'success' qubit.

    For example, if Xi_l=2, |p>  will end up in the state
        |p>|succ_p> = A * ( |00>|1> + |01>|1> + |10>|1> + |11>|0> )

    Note, the |unary_ancilla>, |less_than_equal_ancilla>, and |rot_ancilla> registers all start and end in the |0> state. |phi> should be a phase gradient state, |rot_data> should be in a state corresponding to the binary representation of the angle used for amplitude amplification, and |Xi_l> should be in a state corresponding to the binary representation of Xi_l.

    References:
        [1] https://arxiv.org/abs/2011.03494 
    '''

    br: int # number of precision bits for aa rotation
    bphi:int # number of phase gradient bits
    n_Xi: int # number of Xi bits = int(np.ceil(np.log2(max(Xi_vals)))) 

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(
            unary_ancilla=self.n_Xi,
            Xi_l=self.n_Xi,
            p=self.n_Xi, # p iterates over range(Xi^(ell))
            rot_data=self.br,
            phi=self.bphi,
            less_than_equal_ancilla=1,
            rot_ancilla=1,
            succ_p=1
        )

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        index_reg, Xi_reg = quregs['p'], quregs['Xi_l']
        unary_ancilla_reg, less_than_equal_ancilla, rot_ancilla = quregs['unary_ancilla'], quregs['less_than_equal_ancilla'], quregs['rot_ancilla']
        success_ancilla = quregs['succ_p']
        phase_gradient_state, rot_data = quregs['phi'], quregs['rot_data']
        
        # i. n_Xi-1 Toffolis (and Clifford gates) to match binary rep of Xi. This effectively rounds up to the next power of 2 so we can create a uniform superposition using only Hadamards.
        yield BinaryToUnaryBits.make_on(binary=Xi_reg, unary=unary_ancilla_reg)

        # ii. n_Xi Hadamards on index reg controlled by unary register generated above
        ## TODO: catalytically controlled Hadamard (Fig 17 of ref)
        yield [
            op.controlled_by(unary_ancilla_reg[i]) for i,op in enumerate(cirq.H.on_each(*index_reg))
        ]

        # iii. Inequality test between Xi and index register (p)
        yield LessThanEqual(x_bitsize=self.n_Xi, y_bitsize=self.n_Xi).on(
            *index_reg, *Xi_reg, *less_than_equal_ancilla
        )

        # iv. Rotate ancilla qubit based on rot angle output by QROM
        yield cirq.Rx(rads=np.pi/2).on(*rot_ancilla)
        yield PhaseGradientZRotation(br=self.br,bphi=self.bphi,do_negative_z_rotation=True).on_registers(rotation_target=rot_ancilla,phi=phase_gradient_state,angle=rot_data)
        yield cirq.Rx(rads=-np.pi/2).on(*rot_ancilla)

        # v. Reflection on result of inequality test and ancilla
        yield cirq.CZ(*less_than_equal_ancilla,*rot_ancilla)

        # vi. Invert rotation
        yield cirq.Rx(rads=np.pi/2).on(*rot_ancilla)
        yield PhaseGradientZRotation(br=self.br,bphi=self.bphi,do_negative_z_rotation=False).on_registers(rotation_target=rot_ancilla,phi=phase_gradient_state,angle=rot_data)
        yield cirq.Rx(rads=-np.pi/2).on(*rot_ancilla)
        ## and invert inequality test
        yield LessThanEqual(x_bitsize=self.n_Xi, y_bitsize=self.n_Xi).on(
            *index_reg, *Xi_reg, *less_than_equal_ancilla
        )

        # vii. Repeat controlled Hadamards
        yield [
            op.controlled_by(unary_ancilla_reg[i]) for i,op in enumerate(cirq.H.on_each(*index_reg))
        ]

        # viii. Reflect about zero state on n_Xi+1 qubits (index reg p and ancilla)
        if self.n_Xi == 1:
            yield cirq.CZ(*index_reg,*rot_ancilla)
        else:
            yield MultiControlPauli(cvs=(1,)*self.n_Xi,target_gate=cirq.Z).on_registers(controls=index_reg,target=rot_ancilla)

        # ix. Repeat controlled Hadamards. 
        yield [
            op.controlled_by(unary_ancilla_reg[i]) for i,op in enumerate(cirq.H.on_each(*index_reg))
        ]

        # x. Repeat inequality test
        yield LessThanEqual(x_bitsize=self.n_Xi, y_bitsize=self.n_Xi).on(
            *index_reg, *Xi_reg, *less_than_equal_ancilla
        )

        ## success flagged on inequality test
        yield cirq.CNOT(*less_than_equal_ancilla,*success_ancilla)

        # Invert inequality test
        yield LessThanEqual(x_bitsize=self.n_Xi, y_bitsize=self.n_Xi).on(
            *index_reg, *Xi_reg, *less_than_equal_ancilla
        )

        ## Invert binary to unary conversion with Cliffords
        yield BinaryToUnaryBits.make_on(binary=Xi_reg, unary=unary_ancilla_reg)**(-1)