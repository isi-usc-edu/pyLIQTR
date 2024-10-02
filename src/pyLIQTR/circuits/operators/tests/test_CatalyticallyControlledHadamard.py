"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq
import numpy as np

from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim
from qualtran.cirq_interop.testing import GateHelper
from pyLIQTR.utils.global_ancilla_manager import gam as gam
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.CatalyticallyControlledHadamard import CatalyticallyControlledHadamard
from pyLIQTR.utils.resource_analysis import estimate_resources

class TestCatalyticallyControlledHadamard:

    def test_CatalyticallyControlledHadamard(self):
        '''
        Tests the gate carries out the expected operation by comparing simulation output with the expected output for different input values.
        '''
        gate = CatalyticallyControlledHadamard(prepare_catalytic_state=True)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # circuit
        circuit = cirq.Circuit(operation)
        result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=helper.all_qubits,initial_state=0b1000)
        output_state = result.final_state_vector
        # set up expected output state
        expected_out = np.zeros(2**4,dtype=np.complex128)
        expected_out[10] = 1/(2*np.sqrt(2))*(1+1j)
        expected_out[14] = 1/(2*np.sqrt(2))*(1+1j)
        expected_out[8] = 1/2
        expected_out[12] = 1/2
        # assert close up to global phase
        cirq.testing.assert_allclose_up_to_global_phase(expected_out, output_state, atol=1e-6)

        # check when control turned off
        result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=helper.all_qubits,initial_state=0b0000)
        output_state = result.final_state_vector
        # set up expected output state
        expected_out = np.zeros(2**4,dtype=np.complex128)
        expected_out[0] = 1/np.sqrt(2)
        expected_out[2] = 1/2 * (1+1j)
        # assert close up to global phase
        cirq.testing.assert_allclose_up_to_global_phase(expected_out, output_state, atol=1e-6)
        

    def test_CatalyticallyControlledHadamard_inverse(self):
        '''
        Tests the gate inverse correctly uncomputes by checking all qubits return to initial state when gate*(gate)**(-1) is applied.
        '''
        # initialilze operator
        gate = CatalyticallyControlledHadamard(prepare_catalytic_state=True)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # circuit
        circuit = cirq.Circuit(operation)
        circuit.append(operation**(-1))
        # output states equal input which is |0>
        # verify simulated results
        qubits = circuit.all_qubits()
        init_state = [0]*len(qubits)
        assert_circuit_inp_out_cirqsim(circuit,qubit_order=[*qubits],inputs=init_state,outputs=init_state)

    def test_CatalyticallyControlledHadamard_decomposes(self):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = CatalyticallyControlledHadamard(prepare_catalytic_state=True)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_CatalyticallyControlledHadamard_qasm(self):
        '''
        Tests qasm printing functionality.
        '''
        context = cirq.DecompositionContext(gam)
        gate = CatalyticallyControlledHadamard(prepare_catalytic_state=True)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None
        for line in qasm:
            pass

    def test_CatalyticallyControlledHadamard_resources(self):
        '''
        Tests resources are counted correctly. Assumes cirq default decomposition for controlled Hadamards.
        '''
        gate = CatalyticallyControlledHadamard(prepare_catalytic_state=True)
        resources = estimate_resources(gate)

        cliffs_for_zero_controls = 2*3
        cliffs_for_and = 9
        cliffs_for_and_adj = 4
        cliffs_for_cat_state = 1
        other_cliffs = 12

        assert resources['T'] == 4+1
        assert resources['Clifford'] == cliffs_for_zero_controls+cliffs_for_and+cliffs_for_and_adj+cliffs_for_cat_state+other_cliffs
        assert resources['LogicalQubits'] == 3
        assert gate.t_complexity() == gate.decompose_bloq().t_complexity()