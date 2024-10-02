"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq

from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim, GateHelper
from pyLIQTR.utils.global_ancilla_manager import gam as gam
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.FlagZeroAsFailure import FlagZeroAsFailure
from pyLIQTR.utils.resource_analysis import estimate_resources

class TestFlagZeroAsFailure:

    def test_FlagZeroAsFailure(self):
        '''
        Tests the gate carries out the expected operation by comparing simulation output with the expected output for different input values.
        '''
        bitsize=2
        gate = FlagZeroAsFailure(bitsize=bitsize)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # circuit
        circuit = cirq.Circuit(operation)
        # check simulation results for 4 cases: x dim has -0, y dim has -0, z dim has -0, no dim has -0
        input_states = [[1,0,0,1,1,1,0,0,1],[0,1,1,1,0,0,1,0,1],[1,1,0,1,0,1,1,0,0],[0,1,0,0,1,1,1,0,1]]
        output_states = [[1,0,0,1,0,0,1,1],[0,1,1,0,1,0,1,1],[0,1,1,0,0,1,0,1],[0,0,1,0,0,0,0,0]]
        for i,state in enumerate(input_states):
            full_input_state = state + [0]*(5+3*(bitsize-1))
            full_output_state = state + output_states[i]
            assert_circuit_inp_out_cirqsim(circuit,qubit_order=[*helper.quregs['nu'].flatten(),*helper.quregs['and_ancilla'].flatten(),*helper.quregs['flag_dim'].flatten(),*helper.quregs['flag_ancilla'],*helper.quregs['flag_minus_zero']],inputs=full_input_state,outputs=full_output_state)   
                

    def test_FlagZeroAsFailure_inverse(self):
        '''
        Tests the gate inverse correctly uncomputes by checking all qubits return to initial state when gate*(gate)**(-1) is applied.
        '''
        # initialilze operator
        gate = FlagZeroAsFailure(bitsize=2)
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

    def test_FlagZeroAsFailure_decomposes(self):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = FlagZeroAsFailure(bitsize=3)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_FlagZeroAsFailure_qasm(self):
        '''
        Tests qasm printing functionality.
        '''
        context = cirq.DecompositionContext(gam)
        bitsize = 3
        gate = FlagZeroAsFailure(bitsize=bitsize)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None
        for line in qasm:
            pass

    def test_FlagZeroAsFailure_resources(self):
        '''
        Tests resources are counted correctly.
        '''
        bitsize=3
        gate = FlagZeroAsFailure(bitsize=bitsize)
        resources = estimate_resources(gate)

        cliffs_for_zero_controls = 8+3*2*bitsize
        cliffs_for_and = 9
        num_ands = 2+3*bitsize

        assert resources['T'] == 4*(3*bitsize+2) #matches paper
        assert resources['Clifford'] == 2 + num_ands*cliffs_for_and + cliffs_for_zero_controls
        assert resources['LogicalQubits'] == 3*(2*bitsize + 1) + 2
        assert gate.t_complexity() == gate.decompose_bloq().t_complexity()