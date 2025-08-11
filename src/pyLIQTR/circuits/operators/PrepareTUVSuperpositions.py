"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
from functools import cached_property
from typing import Set

import cirq
import numpy as np
from numpy.typing import NDArray

from qualtran import (
    GateWithRegisters,
    QAny,
    QBit,
    Register,
    Signature,
    Side,
)
from qualtran.bloqs.basic_gates import Hadamard, SGate, XGate
from qualtran.bloqs.mcmt import And
from qualtran.bloqs.arithmetic.comparison import LessThanConstant
from qualtran.symbolics.math_funcs import bit_length

from pyLIQTR.circuits.operators.FlaggedPrepareUniformSuperposition import FlaggedPrepareUniformSuperposition
from pyLIQTR.circuits.operators.PhaseGradientRotation import PhaseGradientZRotation

class PrepareTUVSuperpositions(GateWithRegisters):
    """Prepare the superposition over registers selecting between T, U and V.

    Prepares registers a (tuv) and i (uv/superposition_state).

    Registers:

    .. line-block::
        tuv: a single qubit rotated to appropriately weight T (|0>) and U or V (|1>).
        uv: a single qubit flagging V (|0>).
        rot_ancilla: a single qubit rotated as part of the uniform superposition subroutine.
        flag_prep_success: a single qubit flagging the success of the uniform superposition preparation.
        flag_inequality: a single qubit flagging the result of the inequality test.
        superposition_state: Register that uniform superposition is prepared on.
        phase_gradient_state: Register containing the phase gradient state.

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_
        pg 17-18, Eq. 61
    """

    def __init__(self, num_bits_t:int, eta:int, lambda_zeta:int,bphi:int=None, theta:float=np.pi):
        self.num_bits_t = num_bits_t
        self.eta = eta
        self.lambda_zeta = lambda_zeta
        if bphi is None:
            self.bphi = self.num_bits_t
        else:
            self.bphi = bphi
        self.theta = theta
        self.n_eta_zeta = bit_length(self.eta + 2 * self.lambda_zeta - 1)

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("tuv", QBit()),
                Register("uv", QBit(),side=Side.RIGHT),
                Register("rot_ancilla", QBit()),
                Register("flag_prep_success", QBit()),
                Register("flag_inequality", QBit()),
                Register("superposition_state", QAny(bitsize=self.n_eta_zeta)),
                Register("phase_gradient_state", QAny(bitsize=self.bphi)),
            ]
        )

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        tuv = quregs['tuv']
        uv = quregs['uv']
        rot_ancilla, flag_prep_success = quregs['rot_ancilla'], quregs['flag_prep_success']
        flag_inequality = quregs['flag_inequality']
        superposition_state = quregs['superposition_state']
        phase_gradient_state = quregs['phase_gradient_state']

        
        # rotate tuv into the state cos(theta)|0> + sin(theta)|1>
        yield Hadamard().on(*tuv)
        
        yield PhaseGradientZRotation(br=self.num_bits_t,bphi=self.bphi,do_negative_z_rotation=True,classical_angle=True,angle=self.theta).on_registers(rotation_target=tuv,phi=phase_gradient_state)

        yield [Hadamard().on(*tuv), SGate().on(*tuv)]
        # tuv = |0> indicates T, |1> indicates U/V

        less_than_ancilla = context.qubit_manager.qalloc(1)

        # prepare superposition state and use inequalty test to weight U and V (register i in reference Eq 48)
        yield FlaggedPrepareUniformSuperposition(d=self.eta + 2 * self.lambda_zeta).on_registers(target=superposition_state,less_than_ancilla=less_than_ancilla,rot_ancilla=rot_ancilla,success=flag_prep_success)

        context.qubit_manager.qfree(less_than_ancilla)

        yield LessThanConstant(self.n_eta_zeta,less_than_val=self.eta).on_registers(x=superposition_state,target=flag_inequality)
        # when flag_inequality is 1, state is < eta --> corresponds to V

        # And controlled on tuv, flag_inequality. Flags when V should be selected (ie if uv=|0>). See paragraph above Eq 57.
        yield And(cv1=1,cv2=1).on_registers(ctrl=[tuv,flag_inequality],target=uv)
        yield XGate().on(*uv)
        # uv = |1> indicates U/T, uv = |0> indicates V


    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(Hadamard(),2),(SGate(),1),(PhaseGradientZRotation(br=self.num_bits_t,bphi=self.bphi,do_negative_z_rotation=True,classical_angle=True,angle=self.theta),1),(FlaggedPrepareUniformSuperposition(d=self.eta + 2 * self.lambda_zeta),1),(LessThanConstant(self.n_eta_zeta,less_than_val=self.eta),1),(And(cv1=1,cv2=1),1),(XGate(),1)}