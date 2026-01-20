"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
import attrs
import numpy as np
from  pyLIQTR.utils.global_ancilla_manager import gam

from functools import cached_property
from numpy.typing import NDArray
from typing import List, Set

from qualtran import GateWithRegisters, Signature, QFxp, QUInt
from qualtran.bloqs.mcmt import MultiTargetCNOT
from qualtran.bloqs.arithmetic import Add, AddK
from qualtran.bloqs.basic_gates import XGate


def approx_angle_with_br_bits(angle:float,br:int=8):
    # normalize angle
    angle_norm = angle / (2*np.pi) % 1
    # approximate to br bits and express in binary
    binary_angle = QFxp(br, br, signed=False).to_bits(angle_norm)
    return binary_angle

@attrs.frozen
class PhaseGradientZRotation(GateWithRegisters):
    '''
    Uses addition into a :math:`b_\\phi`-qubit phase gradient state :math:`|\\phi\\rangle` to apply a single qubit Z rotation to |rotation_target> where the rotation angle is provided on a :math:`b_r`-qubit register |angle>.

    Specifically, for the exact rotation angle :math:`\\theta \in [0,2\\pi]` and :math:`\\theta_N = \\theta/(2\\pi)\\mod{1}`, 
    |angle> will encode the :math:`b_r`-bit approximation :math:`\\theta_N = \\sum_{b=0}^{b_r-1} \\theta_b/2^{(1+b)}` where each :math:`\\theta_b` value is stored on the :math:`b`-th qubit of |angle>. In other words, |angle> encodes the two's complement of :math:`\\theta_N`.

    For example, let :math:`k = \\sum_{b=0}^{b_r-1} \\theta_b 2^{(1+b)}` be the integer representation of the normalized rotation angle, then the rotation applied to |rotation_target> is :math:`\\exp(iZt)`, where :math:`t = \\pi k/(2^{(b_r-1)})`. ie, let |rotation_target> = A|0> + B|1>. Then, this gate implements

    .. math::
        U(A|0\\rangle + B|1\\rangle)|k\\rangle|\\phi\\rangle \\rightarrow \\left( A\\exp(i2\\pi k/2^{b_r})|0\\rangle + B\\exp(-i2\\pi k/2^{b_r})|1\\rangle \\right)|k\\rangle|\\phi\\rangle 
     
    using :math:`4(b_r-1)` T-gates which is the cost of the addition.

    The overall sign in the exponent is set by the argument do_negative_z_rotation as follows,

    .. line-block::
        do_negative_z_rotation=True -> exp(-iZt)
        do_negative_z_rotation=False -> exp(iZt)
        Note, cirq.Rz(t) = exp(-iZt/2).

    References:
        `Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization <https://arxiv.org/abs/2007.07391>`_
        Appendix A

    :param int br: number of precision bits for rotation angle
    :param int bphi: bitsize of phase gradient state
    :param bool do_negative_z_rotation: set to True for exp(-iZt), false for exp(iZt) -- determines value of rotation_target to control on.
    :param bool classical_angle: set to True to provide the angle classically rather than previously loaded on a quantum register. Default is False.
    :param float angle: The value of the rotation angle in radians. Only needed when classical_angle=True.
    '''

    br: int 
    bphi: int 
    do_negative_z_rotation: bool = False 
    classical_angle: bool = False
    angle: float = None

    @classmethod
    def make_on(cls,angle_data_qubits:List[cirq.Qid],phase_gradient_state:List[cirq.Qid],rotation_target_qubit:List[cirq.Qid],do_negative_z_rotation: bool = False):
        '''
        Automatically determines br and bphi from number of qubits passed in. Assumes angle is preloaded on angle_data_qubits.
        '''
        assert(len(angle_data_qubits)<=len(phase_gradient_state))

        return PhaseGradientZRotation(br=len(angle_data_qubits),bphi=len(phase_gradient_state),do_negative_z_rotation=do_negative_z_rotation).on_registers(rotation_target=rotation_target_qubit,phi=phase_gradient_state,angle=angle_data_qubits)

    @cached_property
    def signature(self) -> Signature:
        if not self.classical_angle:
            return Signature.build(
                rotation_target=1,
                phi=self.bphi,
                angle=self.br,
            )
        else:
            return Signature.build(
                rotation_target=1,
                phi=self.bphi
            )

    def approx_angle_as_br_int(self):
        '''
        Returns the classicaly provided angle as an integer approximated with br bits.
        '''
        assert self.angle is not None
        # normalize angle
        angle_norm = self.angle / (2*np.pi) % 1
        # approximate to br bits and express in binary
        binary_angle = []
        for _ in range(self.br):
            angle_norm = angle_norm * 2
            out_bit = np.floor(angle_norm)
            angle_norm = angle_norm - out_bit
            binary_angle.append(int(out_bit))
        integer_angle = QUInt(self.br).from_bits(binary_angle)
        return integer_angle

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext=None,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        phase_gradient_state = quregs['phi']
        rotation_target = quregs['rotation_target']
        angle = quregs.get('angle',())

        if self.do_negative_z_rotation:
            yield XGate().on(*rotation_target)

        # apply CNOT to phi register so that |1> part of rotation_target will store subtraction (for self.do_negative_z_rotation=False, |0> for True)
        yield MultiTargetCNOT(self.bphi).on_registers(control=rotation_target,targets=phase_gradient_state)

        # in place addition -- result is output on phase_gradient_state
        # TODO: optimize addition, see: https://arxiv.org/abs/2007.07391 Appendix D2 Figure 18 for details
        if not self.classical_angle:
            yield Add(a_dtype=QUInt(self.br),b_dtype=QUInt(self.bphi)).on_registers(a=angle,b=phase_gradient_state)
        else:
            int_angle = self.approx_angle_as_br_int()
            yield AddK(bitsize=self.bphi,k=int_angle).on_registers(x=phase_gradient_state)

        # apply CNOT to register storing phi+angle
        yield MultiTargetCNOT(self.bphi).on_registers(control=rotation_target,targets=phase_gradient_state)

        if self.do_negative_z_rotation:
            yield XGate().on(*rotation_target)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        if self.do_negative_z_rotation: X_counts = {(XGate(),2)}
        else: X_counts = set()

        if self.classical_angle: adder = AddK(dtype=QUInt(self.bphi),k=self.approx_angle_as_br_int())
        else: adder = Add(a_dtype=QUInt(self.br),b_dtype=QUInt(self.bphi))

        return {(MultiTargetCNOT(self.bphi),2),(adder,1)} | X_counts