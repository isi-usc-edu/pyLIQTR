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
from  pyLIQTR.utils.global_ancilla_manager import gam

from cirq._compat import cached_property
from numpy.typing import NDArray
from typing import List

from qualtran import GateWithRegisters, Signature
from qualtran.bloqs.multi_control_multi_target_pauli import MultiTargetCNOT
from qualtran.bloqs.arithmetic import Add
from qualtran.cirq_interop.bit_tools import iter_bits_fixed_point

def approx_angle_with_br_bits(angle:float,br:int=8):
    # normalize angle
    angle_norm = angle / (2*np.pi) % 1
    # approximate to br bits and express in binary
    binary_angle = list(iter_bits_fixed_point(angle_norm,width=br,signed=False))
    return binary_angle

@attrs.frozen
class PhaseGradientZRotation(GateWithRegisters):
    '''
    Uses addition into a br-qubit phase gradient state |phi> to apply a single qubit Z rotation to |rotation_target> where the rotation angle is provided on a br-qubit register |angle>.

    Specifically, for the exact rotation angle theta in [0,2*pi] and theta_N = theta/(2*pi) % 1, 
    |angle> will encode the br-bit approximation theta_N = sum_{b=0}^{br-1} theta_b/2^(1+b) where each theta_b value is stored on the bth qubit of |angle>. In other words, |angle> encodes the two's complement of theta_N.

    For example, let k = sum_{b=0}^{br-1} theta_b*2^{1+b} be the integer representation of the normalized rotation angle, then the rotation applied to |rotation_target> is exp(iZt), where t = pi*k/(2^(br-1)). ie, let |rotation_target> = A|0> + B|1>. Then, this gate implements

     U(A|0> + B|1>)|k>|phi> -> ( A*exp(i2pi*k/(2^br))|0> + B*exp(-i2pi*k/(2^br))|1> )|k>|phi> 
     
    using 4*(br-1) T-gates which is the cost of the addition.

    The overall sign in the exponent is set by the argument do_negative_z_rotation as follows,
    do_negative_z_rotation=True -> exp(-iZt)
    do_negative_z_rotation=False -> exp(iZt)
    Note, cirq.Rz(t) = exp(-iZt/2).

    See https://arxiv.org/abs/2007.07391 Appendix A for details.
    '''

    br: int # number of precision bits for rotation angle
    bphi: int # bitsize of phase gradient state
    do_negative_z_rotation: bool = False #set to true for exp(-iZt), false for exp(iZt) -- determines value of rotation_target to control on

    @classmethod
    def make_on(cls,angle_data_qubits:List[cirq.Qid],phase_gradient_state:List[cirq.Qid],rotation_target_qubit:List[cirq.Qid],do_negative_z_rotation: bool = False):
        '''
        Automatically determines br and bphi from number of qubits passed in
        '''
        assert(len(angle_data_qubits)<=len(phase_gradient_state))

        return PhaseGradientZRotation(br=len(angle_data_qubits),bphi=len(phase_gradient_state),do_negative_z_rotation=do_negative_z_rotation).on_registers(rotation_target=rotation_target_qubit,phi=phase_gradient_state,angle=angle_data_qubits)

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(
            rotation_target=1,
            phi=self.bphi,
            angle=self.br,
        ) 

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext=None,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        # set decomp context to use pyLIQTR global ancilla manager
        if context is None:
            context=cirq.DecompositionContext(gam)

        phase_gradient_state = quregs['phi']
        rotation_target = quregs['rotation_target']
        angle = quregs['angle']
        zero_padding = context.qubit_manager.qalloc(self.bphi - self.br)

        if self.do_negative_z_rotation:
            yield cirq.X.on(*rotation_target)

        # apply CNOT to phi register so that |1> part of rotation_target will store subtraction (for self.do_negative_z_rotation=False, |0> for True)
        yield MultiTargetCNOT(self.bphi).on_registers(control=rotation_target,targets=phase_gradient_state)

        # in place addition -- result is output on phase_gradient_state
        # TODO: optimize addition, see: https://arxiv.org/abs/2007.07391 Appendix D2 Figure 18 for details
        yield Add(bitsize=self.bphi).on_registers(a=list(zero_padding)+list(angle),b=phase_gradient_state)

        # apply CNOT to register storing phi+angle
        yield MultiTargetCNOT(self.bphi).on_registers(control=rotation_target,targets=phase_gradient_state)

        if self.do_negative_z_rotation:
            yield cirq.X.on(*rotation_target)

        context.qubit_manager.qfree(zero_padding)

    def _num_qubits_(self):
        n_ancilla = self.bphi - self.br
        # TODO count ancilla used in Add and CNOT
        return sum(reg.total_bits() for reg in self.signature)+n_ancilla

    def _qid_shape_(self):
        num_qb_in_sig = sum(reg.total_bits() for reg in self.signature)
        return (2,)*num_qb_in_sig