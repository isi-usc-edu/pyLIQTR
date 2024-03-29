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
from qualtran import GateWithRegisters, Signature
import numpy as np
from cirq._compat import cached_property
from numpy.typing import NDArray
from pyLIQTR.circuits.operators.PhaseGradientRotation import PhaseGradientZRotation

@attrs.frozen
class ProductPauliExponentials(GateWithRegisters):
    '''
    Implements

    U|theta_p>|phi>|p>|q> -> |theta_p>|phi>exp(-i*theta_p*Y_p*X_q)exp(i*theta_p*X_p*Y_q)|p>|q>

    where X_p (Y_q) indicates the Pauli operator acting on qubit p (q), |theta_p> is a br-qubit register storing the angle theta_p using br bits of precision, |phi> is a bphi-qubit register prepared as a phase gradient state, and |p> and |q> are the target qubits. This gate is used to rotate the Majorana operator into a new basis in the DoubleFactorized block encoding corresponding to eq 62 in ref [1] (Note eq. 62 is the conjugate transpose so the signs are swapped). The implementation transforms each exponent into a single qubit Z rotation surrounded by Clifford gates using the staircase algorithm as described in ref [2]. The Z rotation is then carried out using addition into a phase gradient state.

    References:
    [1] https://arxiv.org/pdf/2007.14460.pdf 
    [2] https://arxiv.org/pdf/2305.04807.pdf
    '''
    br: int # number of precision bits for rotation angle
    bphi: int # number of qubits [hase gradient state is prepared on
    uncompute: bool = False # set to True if uncomputing the gate. Amounts to swapping the signs on the angle in the rotation

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(
            angle_data=self.br,
            target=2,
            phi=self.bphi
        )

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:

        angle_reg = quregs['angle_data']
        target = quregs['target']
        phi = quregs['phi']
        target0 = target[0]
        target1 = target[1]

        pre_XY_exp = [
            cirq.H(target0),
            cirq.H(target1),
            cirq.S(target1),
            cirq.CNOT(target0,target1)
        ]

        between_exp = [
            cirq.CNOT(target0,target1),
            cirq.S(target1)**(-1),
            cirq.S(target0),
            cirq.CNOT(target0,target1)
        ]

        post_YX_exp = [
            cirq.CNOT(target0,target1),
            cirq.S(target0)**(-1),
            cirq.H(target1),
            cirq.H(target0)
        ]

        yield pre_XY_exp

        if not self.uncompute:
            yield PhaseGradientZRotation(self.br,bphi=self.bphi,do_negative_z_rotation=True).on_registers(rotation_target=target1,phi=phi,angle=angle_reg)
        else:
            yield PhaseGradientZRotation(self.br,bphi=self.bphi,do_negative_z_rotation=False).on_registers(rotation_target=target1,phi=phi,angle=angle_reg)

        yield between_exp

        if not self.uncompute:
            yield PhaseGradientZRotation(self.br,bphi=self.bphi,do_negative_z_rotation=False).on_registers(rotation_target=target1,phi=phi,angle=angle_reg)
        else:
            yield PhaseGradientZRotation(self.br,bphi=self.bphi,do_negative_z_rotation=True).on_registers(rotation_target=target1,phi=phi,angle=angle_reg)

        yield post_YX_exp