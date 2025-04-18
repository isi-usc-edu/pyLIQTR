"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq
import numpy as np

from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim, GateHelper
from qualtran.cirq_interop.bit_tools import iter_bits_twos_complement
from  pyLIQTR.utils.global_ancilla_manager import gam as gam
from  pyLIQTR.utils.printing import openqasm
from pyLIQTR.utils.resource_analysis import estimate_resources
from pyLIQTR.utils.circuit_decomposition import decompose_once, circuit_decompose_multi
from pyLIQTR.circuits.operators.PhaseGradientRotation import approx_angle_with_br_bits, PhaseGradientZRotation
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from pyLIQTR.circuits.operators.tests.utils import get_partial_state,strip_bloqs,ignore_bookkeeping

class TestPhaseGradientZRotation:

    @pytest.mark.parametrize("test_angle,br,bphi,negative_Z_rotation,classical_angle",[(np.pi/9,2,5,False,False),(np.pi/9,4,4,False,True),(-1.5*np.pi,2,5,True,False),(-1.5*np.pi,4,4,True,True)])
    def test_PhaseGradientZRotation(self,br,bphi,test_angle,negative_Z_rotation,classical_angle):
        '''
        Tests the gate carries out the expected operation by comparing simulation output with the expected output for different input values.
        '''
        # create registers
        target = [cirq.NamedQubit('rotation_target')]
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
        rot_gate = PhaseGradientZRotation(br=br,bphi=bphi,do_negative_z_rotation=negative_Z_rotation,classical_angle=classical_angle,angle=test_angle)
        helper = GateHelper(rot_gate)
        # remove 'Allocate' and 'Free' bloqs from circuit so simulation works
        decomposed_twice = circuit_decompose_multi(helper.circuit,2)
        stripped_circuit = cirq.Circuit([strip_bloqs(item) for item in decomposed_twice])
        ancilla = [bit for bit in decomposed_twice.all_qubits() if hasattr(bit,'prefix')]
        circuit.append(stripped_circuit)

        # simulate
        res = cirq.Simulator(dtype=np.complex128).simulate(circuit,qubit_order=[*angle,*phi,*target,*ancilla])
        # isolate state of target register just before phase gradient rotation is applied
        state_vector_init = cirq.final_state_vector(circuit[0:1],qubit_order=[*angle,*phi,*target,*ancilla],dtype=np.complex128)
        partial_state_target_init = get_partial_state(state_vector_init,keep_indices=[br+bphi])
        # isolate final state of target
        partial_state_target = get_partial_state(res.final_state_vector,keep_indices=[br+bphi])
        # isolate final state of phase gradient register
        partial_state_phi = get_partial_state(res.final_state_vector,keep_indices=np.arange(br,br+bphi))

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

    @pytest.mark.parametrize("test_angle,br,bphi,negative_Z_rotation,classical_angle",[(np.pi/9,2,5,False,False),(-1.5*np.pi,4,4,True,False),(np.pi/9,3,5,False,True)])
    def test_PhaseGradientZRotation_inverse(self,br,bphi,test_angle,negative_Z_rotation,classical_angle):
        '''
        Tests the gate inverse correctly uncomputes by checking all qubits return to initial state when gate*(gate)**(-1) is applied.
        '''
        # create registers
        target = [cirq.NamedQubit('rotation_target')]
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
        rot_gate = PhaseGradientZRotation(br=br,bphi=bphi,do_negative_z_rotation=negative_Z_rotation,classical_angle=classical_angle,angle=test_angle)
        helper = GateHelper(rot_gate)
        decomposed_twice = circuit_decompose_multi(helper.circuit,2)
        stripped_circuit = stripped_circuit = cirq.Circuit([strip_bloqs(item) for item in decomposed_twice])
        ancilla = [bit for bit in decomposed_twice.all_qubits() if hasattr(bit,'prefix')]
        circuit.append(stripped_circuit)

        circuit.append(cirq.inverse(circuit))
        # output states equal input which is |0>
        # verify simulated results
        init_state = [0]*br+[0]*bphi+[0] + [0]*len(ancilla)
        assert_circuit_inp_out_cirqsim(circuit,qubit_order=[*angle,*phi,*target,*ancilla],inputs=init_state,outputs=init_state)

    @pytest.mark.parametrize("test_angle,br,bphi,negative_Z_rotation,classical_angle",[(np.pi/9,2,5,False,False),(-1.5*np.pi,8,8,True,False),(np.pi/9,2,5,False,True)])
    def test_PhaseGradientZRotation_decomposes(self,br,bphi,test_angle,negative_Z_rotation,classical_angle):
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
        rot_gate = PhaseGradientZRotation(br=br,bphi=bphi,do_negative_z_rotation=negative_Z_rotation,classical_angle=classical_angle,angle=test_angle)
        operation = rot_gate.on_registers(angle=angle,phi=phi,rotation_target=target)
        circuit.append(operation)
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(circuit)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    @pytest.mark.parametrize("test_angle,br,bphi,negative_Z_rotation,classical_angle",[(np.pi/9,2,5,False,False),(-1.5*np.pi,8,8,True,False),(np.pi/9,2,5,False,True)])
    def test_PhaseGradientZRotation_qasm(self,br,bphi,test_angle,negative_Z_rotation,classical_angle):
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
        rot_gate = PhaseGradientZRotation(br=br,bphi=bphi,do_negative_z_rotation=negative_Z_rotation,classical_angle=classical_angle,angle=test_angle)
        operation = rot_gate.on_registers(angle=angle,phi=phi,rotation_target=target)
        circuit.append(operation)

        qasm = openqasm(circuit,rotation_allowed=True,context=context)
        assert qasm is not None

    @pytest.mark.parametrize("test_angle,br,bphi,negative_Z_rotation,classical_angle",[(np.pi/9,2,5,False,False),(-1.5*np.pi,8,8,True,False),(np.pi/9,2,5,False,True),(-1.5*np.pi,8,8,True,True)])
    def test_PhaseGradientZRotation_resources(self,br,bphi,test_angle,negative_Z_rotation,classical_angle):
        '''
        Tests resources are counted correctly.
        '''
        gate = PhaseGradientZRotation(br=br,bphi=bphi,do_negative_z_rotation=negative_Z_rotation,classical_angle=classical_angle,angle=test_angle)
        resources = estimate_resources(gate)

        # One extra Toffoli for classical_angle=False since not using AddIntoPhaseGradWithDecompose (due to different bitsizes). For classical_angle=True, optimized addition not implemented yet.
        assert resources['T'] == 4*(bphi-1)
        assert gate.t_complexity() == gate.decompose_bloq().t_complexity()

    @pytest.mark.parametrize("test_angle,br,bphi,negative_Z_rotation,classical_angle",[(np.pi/9,2,5,False,False),(-1.5*np.pi,8,8,True,False),(np.pi/9,2,5,False,True)])
    def test_PhaseGradientZRotation_callgraph(self,br,bphi,test_angle,negative_Z_rotation,classical_angle):
        '''
        Tests ability to generate call_graph and that call_graph counts match decomposition counts.
        '''
        gate = PhaseGradientZRotation(br=br,bphi=bphi,do_negative_z_rotation=negative_Z_rotation,classical_angle=classical_angle,angle=test_angle)
        call_graph, _ = gate.call_graph()

        call_graph_counts = gate.bloq_counts(generalizer=ignore_bookkeeping)
        cbloq = gate.decompose_bloq()
        decomp_counts = cbloq.bloq_counts(generalizer=ignore_bookkeeping)

        assert call_graph_counts == decomp_counts