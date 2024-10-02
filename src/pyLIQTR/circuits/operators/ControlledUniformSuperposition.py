"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
from typing import Tuple

import attr
import cirq
import numpy as np
from functools import cached_property
from numpy.typing import NDArray

from qualtran import GateWithRegisters, Signature
from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd
from qualtran.bloqs.arithmetic import LessThanEqual
from qualtran.bloqs.mcmt.multi_control_multi_target_pauli import MultiControlPauli

from pyLIQTR.circuits.operators.BinaryToUnary import BinaryToUnaryBits
from pyLIQTR.circuits.operators.PhaseGradientRotation import PhaseGradientZRotation


@attr.frozen
class ControlledPrepareUniformSuperposition(GateWithRegisters):
    '''
    Prepares the |p> register in the equal superpostion state

    .. math:: 

        \\frac{1}{\\sqrt{2^{n_\\Xi}}} \\left( \\sum_{i=0}^{\\Xi_\\ell} |i>|1> + \\sum_{i=\\Xi_\\ell+1}^{2^{n_\\Xi}} |i>|0>  \\right)

    where :math:`n_\\Xi = ⎡\\log_2(\\Xi_\\ell+1)⎤`, and the second register is a 'success' qubit that flags states satisfying |state> <= |\\Xi_\\ell>.

    For example, if \\Xi_\\ell=2, |p>|succ_p>  will end up in the state

    .. math::
    
        |p>|succ_p> = \\frac{1}{2} * ( |00>|1> + |01>|1> + |10>|1> + |11>|0> )

    Note, the |unary_ancilla>, |less_than_equal_ancilla>, and |rot_ancilla> registers all start and end in the |0> state. |phi> should be a phase gradient state, |rot_data> should be in a state corresponding to the binary representation of the angle used for amplitude amplification, and :math:`( |\\Xi_\\ell\\rangle )` should be in a state corresponding to the binary representation of :math:`(\\Xi_\\ell)`. This corresponds to step 3a on page 52 of the reference.

    References:
        [1] `Even more efficient quantum computations of chemistry through tensor hypercontraction <https://arxiv.org/abs/2011.03494>`_ pg 52

    :param int br: the number of precision qubits for amplitude amplification.
    :param int bphi: the number of qubits in the phase gradient state.
    :param int n_Xi: the number of qubits used to represent :math:`$ \\Xi_\\ell $`.
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
        **quregs: NDArray[cirq.Qid],
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
            yield MultiControlPauli(cvs=(1,)*self.n_Xi,target_gate=cirq.Z).on_registers(controls=[[q] for q in index_reg],target=rot_ancilla)

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