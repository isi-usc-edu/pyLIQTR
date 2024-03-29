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
import pytest
import cirq
import numpy as np

from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim
from qualtran.cirq_interop.bit_tools import iter_bits_twos_complement
from  pyLIQTR.utils.global_ancilla_manager import gam as gam
from  pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.PhaseGradientRotation import approx_angle_with_br_bits, PhaseGradientZRotation
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from pyLIQTR.circuits.operators.tests.utils import get_partial_state

class TestPhaseGradientZRotation:

    @pytest.mark.parametrize("test_angle,br,bphi,negative_Z_rotation",[(np.pi/9,2,5,False),(np.pi/9,8,8,False),(-1.5*np.pi,2,5,True),(-1.5*np.pi,8,8,True)])
    def test_PhaseGradientZRotation(self,br,bphi,test_angle,negative_Z_rotation):
        '''
        Tests the gate carries out the expected operation by comparing simulation output with the expected output for different input values.
        '''
        # create registers
        target = [cirq.NamedQubit('target')]
        angle = cirq.NamedQubit.range(br,prefix='angle')
        phi = cirq.NamedQubit.range(bphi,prefix='phi')

        # approximate angle and get integer representation of it (ell)
        binary_angle = approx_angle_with_br_bits(test_angle,br)
        ell = int(''.join(str(b) for b in binary_angle), 2)

        circuit = cirq.Circuit()
        # prepare angle register
        for i,bit in enumerate(angle):
            if binary_angle[i]:
                circuit.append(cirq.X.on(bit))

        # prepare phase gradient state
        create_phase_grad_state = PhaseGradientState(bitsize=bphi)
        circuit.append(create_phase_grad_state.on(*phi))

        # prepare target in superposition
        circuit.append(cirq.H.on(*target))

        # phase gradient rotation
        rot_gate = PhaseGradientZRotation(br=br,bphi=bphi,do_negative_z_rotation=negative_Z_rotation)
        circuit.append(rot_gate.on_registers(angle=angle,phi=phi,rotation_target=target))

        # isolate state of target register just before phase gradient rotation is applied
        state_vector_init = cirq.final_state_vector(circuit[0:1],qubit_order=[*angle,*phi,*target],dtype=np.complex128)
        partial_state_target_init = get_partial_state(state_vector_init,keep_indices=[-1])
        # isolate final state of target
        partial_state_target = get_partial_state(circuit.final_state_vector(),keep_indices=[-1])
        # isolate final state of phase gradient register
        partial_state_phi = get_partial_state(circuit.final_state_vector(),keep_indices=np.arange(br,br+bphi))

        # expected output for phase gradient state (phi) based on analytic form 
        expected_phi_out=np.round(1/np.sqrt(2**bphi)*np.array([np.exp(-2j*np.pi*(n)/(2**(bphi))) for n in range(2**bphi)]),8)
        # expected output for target is 
        if negative_Z_rotation:
            # equivalent to exp(-i2pi*Z*ell/(2**bphi))|partial_state_target_init>
            expected_target_out = np.array([np.exp(-2j*np.pi*ell/(2**bphi)),np.exp(2j*np.pi*ell/(2**bphi))])*partial_state_target_init
        else:
            # equivalent to exp(i2pi*Z*ell/(2**bphi))|partial_state_target_init>
            expected_target_out = np.array([np.exp(2j*np.pi*ell/(2**bphi)),np.exp(-2j*np.pi*ell/(2**bphi))])*partial_state_target_init

        # assert circuit final state vectors match expected outputs
        assert cirq.allclose_up_to_global_phase(partial_state_target, expected_target_out)
        assert cirq.allclose_up_to_global_phase(partial_state_phi, expected_phi_out)

    @pytest.mark.parametrize("test_angle,br,bphi,negative_Z_rotation",[(np.pi/9,2,5,False),(-1.5*np.pi,8,8,True)])
    def test_PhaseGradientZRotation_inverse(self,br,bphi,test_angle,negative_Z_rotation):
        '''
        Tests the gate inverse correctly uncomputes by checking all qubits return to initial state when gate*(gate)**(-1) is applied.
        '''
        # create registers
        target = [cirq.NamedQubit('target')]
        angle = cirq.NamedQubit.range(br,prefix='angle')
        phi = cirq.NamedQubit.range(bphi,prefix='phi')

        # approximate angle and get integer representation of it (ell)
        binary_angle = approx_angle_with_br_bits(test_angle,br)
        ell = int(''.join(str(b) for b in binary_angle), 2)

        circuit = cirq.Circuit()
        # prepare angle register
        for i,bit in enumerate(angle):
            if binary_angle[i]:
                circuit.append(cirq.X.on(bit))

        # prepare phase gradient state
        create_phase_grad_state = PhaseGradientState(bitsize=bphi)
        circuit.append(create_phase_grad_state.on(*phi))

        # prepare target in superposition
        circuit.append(cirq.H.on(*target))

        # phase gradient rotation
        rot_gate = PhaseGradientZRotation(br=br,bphi=bphi,do_negative_z_rotation=negative_Z_rotation)
        circuit.append(rot_gate.on_registers(angle=angle,phi=phi,rotation_target=target))

        circuit.append(cirq.inverse(circuit))
        # output states equal input which is |0>
        # verify simulated results
        init_state = [0]*br+[0]*bphi+[0]
        assert_circuit_inp_out_cirqsim(circuit,qubit_order=[*angle,*phi,*target],inputs=init_state,outputs=init_state)

    @pytest.mark.parametrize("test_angle,br,bphi,negative_Z_rotation",[(np.pi/9,2,5,False),(-1.5*np.pi,8,8,True)])
    def test_PhaseGradientZRotation_decomposes(self,br,bphi,test_angle,negative_Z_rotation):
        '''
        Tests gate decomposition existence.
        '''
        # create registers
        target = [cirq.NamedQubit('target')]
        angle = cirq.NamedQubit.range(br,prefix='angle')
        phi = cirq.NamedQubit.range(bphi,prefix='phi')

        # approximate angle and get integer representation of it (ell)
        binary_angle = approx_angle_with_br_bits(test_angle,br)
        ell = int(''.join(str(b) for b in binary_angle), 2)

        circuit = cirq.Circuit()
        # prepare angle register
        for i,bit in enumerate(angle):
            if binary_angle[i]:
                circuit.append(cirq.X.on(bit))

        # prepare phase gradient state
        create_phase_grad_state = PhaseGradientState(bitsize=bphi)
        circuit.append(create_phase_grad_state.on(*phi))

        # prepare target in superposition
        circuit.append(cirq.H.on(*target))

        # phase gradient rotation
        rot_gate = PhaseGradientZRotation(br=br,bphi=bphi,do_negative_z_rotation=negative_Z_rotation)
        operation = rot_gate.on_registers(angle=angle,phi=phi,rotation_target=target)
        circuit.append(operation)
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(circuit)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    @pytest.mark.parametrize("test_angle,br,bphi,negative_Z_rotation",[(np.pi/9,2,5,False),(-1.5*np.pi,8,8,True)])
    def test_PhaseGradientZRotation_qasm(self,br,bphi,test_angle,negative_Z_rotation):
        '''
        Tests qasm printing functionality.
        '''
        context = cirq.DecompositionContext(gam)
        # create registers
        target = [cirq.NamedQubit('target')]
        angle = cirq.NamedQubit.range(br,prefix='angle')
        phi = cirq.NamedQubit.range(bphi,prefix='phi')

        # approximate angle and get integer representation of it (ell)
        binary_angle = approx_angle_with_br_bits(test_angle,br)
        ell = int(''.join(str(b) for b in binary_angle), 2)

        circuit = cirq.Circuit()
        # prepare angle register
        for i,bit in enumerate(angle):
            if binary_angle[i]:
                circuit.append(cirq.X.on(bit))

        # prepare phase gradient state
        create_phase_grad_state = PhaseGradientState(bitsize=bphi)
        circuit.append(create_phase_grad_state.on(*phi))

        # prepare target in superposition
        circuit.append(cirq.H.on(*target))

        # phase gradient rotation
        rot_gate = PhaseGradientZRotation(br=br,bphi=bphi,do_negative_z_rotation=negative_Z_rotation)
        operation = rot_gate.on_registers(angle=angle,phi=phi,rotation_target=target)
        circuit.append(operation)

        qasm = openqasm(circuit,rotation_allowed=True,context=context)
        assert qasm is not None