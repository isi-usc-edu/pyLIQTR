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
from  pyLIQTR.utils.global_ancilla_manager import gam as gam
from  pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.PhaseGradientRotation import approx_angle_with_br_bits
from pyLIQTR.circuits.operators.ProductPauliExponentials import ProductPauliExponentials
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from pyLIQTR.circuits.operators.tests.utils import get_partial_state

@pytest.mark.skip(reason='tests are work in progress')
class TestProductPauliExponentials:
    

    @pytest.mark.parametrize("test_angle,br,bphi",[(np.pi/9,2,5),(np.pi/9,8,8),(-1.5*np.pi,2,5),(-1.5*np.pi,8,8)])
    def test_ProductPauliExponetials(self,br,bphi,test_angle):
        '''
        Tests the gate carries out the expected operation when adding/subtracting registers with different bitsizes by comparing simulation output with the expected output for different input values.
        '''
        # create registers
        target = cirq.NamedQubit.range(2,prefix='target')
        angle = cirq.NamedQubit.range(br,prefix='angle')
        phi = cirq.NamedQubit.range(bphi,prefix='phi')

        # approximate angle (factor of 2 since should be Rz(2*test_angle))
        binary_angle = approx_angle_with_br_bits(2*test_angle,br)

        circuit = cirq.Circuit()
        # prepare angle register
        for i,bit in enumerate(angle):
            if binary_angle[i]:
                circuit.append(cirq.X.on(bit))

        # prepare phase gradient state
        create_phase_grad_state = PhaseGradientState(bitsize=bphi)
        circuit.append(create_phase_grad_state.on(*phi))

        # prep target register in random superposition
        circuit.append(cirq.Rx(rads=1.02035).on_each(*target))

        # product pauli exp
        circuit.append(ProductPauliExponentials(br=br,bphi=bphi,uncompute=False).on_registers(angle_data=angle,phi=phi,target=target))

        # isolate state of target register just before phase gradient rotation is applied
        state_vector_init = cirq.final_state_vector(circuit[0:1],qubit_order=[*target,*phi,*angle],dtype=np.complex128)
        partial_state_target_init = get_partial_state(state_vector_init,keep_indices=[0,1])
        # isolate final state of target
        partial_state_target_final = get_partial_state(circuit.final_state_vector(qubit_order=[*target,*phi,*angle]),keep_indices=[0,1])
        # isolate final state of phase gradient register
        partial_state_phi = get_partial_state(circuit.final_state_vector(qubit_order=[*target,*phi,*angle]),keep_indices=np.arange(2,2+bphi))

        # expected output for phase gradient state (phi) based on analytic form 
        expected_phi_out=np.round(1/np.sqrt(2**bphi)*np.array([np.exp(-2j*np.pi*(n)/(2**(bphi))) for n in range(2**bphi)]),8)
        # expected output for target is based on analytic matrix form
        def exp_product_exact(theta):
            return np.array([[1,0,0,0],[0,np.cos(2*theta),-np.sin(2*theta),0],[0,np.sin(2*theta),np.cos(2*theta),0],[0,0,0,1]],dtype=np.complex128)
        analytic_matrix = exp_product_exact(theta=test_angle)
        expected_target_out = analytic_matrix.dot(partial_state_target_init)
        
        # assert circuit final state vectors match expected outputs
        assert cirq.allclose_up_to_global_phase(partial_state_phi, expected_phi_out)
        assert cirq.allclose_up_to_global_phase(partial_state_target_final, expected_target_out)

    @pytest.mark.parametrize("test_angle,br,bphi",[(np.pi/9,2,5),(-1.5*np.pi,8,8)])
    def test_ProductPauliExponentials_uncompute(self,br,bphi,test_angle):
        '''
        Tests the gate inverse correctly uncomputes by checking all qubits return to initial state when gate*(gate)**(-1) is applied.
        '''
        # create registers
        target = cirq.NamedQubit.range(2,prefix='target')
        angle = cirq.NamedQubit.range(br,prefix='angle')
        phi = cirq.NamedQubit.range(bphi,prefix='phi')

        # approximate angle
        binary_angle = approx_angle_with_br_bits(test_angle,br)

        circuit = cirq.Circuit()
        # prepare angle register
        for i,bit in enumerate(angle):
            if binary_angle[i]:
                circuit.append(cirq.X.on(bit))

        # prepare phase gradient state
        create_phase_grad_state = PhaseGradientState(bitsize=bphi)
        circuit.append(create_phase_grad_state.on(*phi))

        # product pauli exp
        circuit.append(ProductPauliExponentials(br=br,bphi=bphi,uncompute=False).on_registers(angle_data=angle,phi=phi,target=target))
        # uncompute
        circuit.append(ProductPauliExponentials(br=br,bphi=bphi,uncompute=True).on_registers(angle_data=angle,phi=phi,target=target))
        # unprepare phase gradient
        circuit.append(cirq.inverse(create_phase_grad_state.on(*phi)))
        # unprepare angle register
        for i,bit in enumerate(angle):
            if binary_angle[i]:
                circuit.append(cirq.X.on(bit))

        # output state equals input which is |0>
        # verify simulated results
        init_state = [0]*br+[0]*bphi+[0]*2
        assert_circuit_inp_out_cirqsim(circuit,qubit_order=[*angle,*phi,*target],inputs=init_state,outputs=init_state)

    @pytest.mark.parametrize("br,bphi",[(2,5),(8,8)])
    def test_PauliProductExponentials_decomposes(self,br,bphi):
        '''
        Tests gate decomposition existence.
        '''
        # create registers
        target = cirq.NamedQubit.range(2,prefix='target')
        angle = cirq.NamedQubit.range(br,prefix='angle')
        phi = cirq.NamedQubit.range(bphi,prefix='phi')
        # product pauli exp
        operation = ProductPauliExponentials(br=br,bphi=bphi,uncompute=False).on_registers(angle_data=angle,phi=phi,target=target)
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_ProductPauliExponentials_qasm(self):
        '''
        Tests qasm printing functionality.
        '''
        context = cirq.DecompositionContext(gam)
        # create registers
        br = 4
        bphi = 4
        target = cirq.NamedQubit.range(2,prefix='target')
        angle = cirq.NamedQubit.range(br,prefix='angle')
        phi = cirq.NamedQubit.range(bphi,prefix='phi')
        # product pauli exp
        operation = ProductPauliExponentials(br=br,bphi=bphi,uncompute=False).on_registers(angle_data=angle,phi=phi,target=target)

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None