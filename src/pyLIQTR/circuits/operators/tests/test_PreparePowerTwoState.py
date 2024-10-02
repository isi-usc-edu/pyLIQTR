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
from pyLIQTR.circuits.operators.PreparePowerTwoState import PreparePowerTwoState
from pyLIQTR.utils.resource_analysis import estimate_resources

class TestPreparePowerTwoState:

    @pytest.mark.parametrize("bitsize",[4,7])
    def test_PreparePowerTwoState(self,bitsize):
        '''
        Tests the gate carries out the expected operation by comparing simulation output with the expected output for different input values.
        '''
        gate = PreparePowerTwoState(bitsize=bitsize,prepare_catalytic_state=True)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # circuit
        circuit = cirq.Circuit(operation)
        result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=helper.all_qubits)
        output_state = cirq.sub_state_vector(result.final_state_vector,keep_indices=[*range(bitsize)])
        # set up expected output state, which is one coefficient entry at each power of two in the state vector
        coeffs = list(reversed([2**((-bitsize+1)/2)*2**(r/2) for r in range(bitsize-1)] + [2**((-bitsize+1)/2)]))
        expected_out = np.array([coeffs[int(np.log2(i))]+0j if i.bit_count()==1 else 0j for i in range(2**bitsize)])
        cirq.testing.assert_allclose_up_to_global_phase(expected_out, output_state, atol=1e-6)
        

    @pytest.mark.parametrize("bitsize",[4,7])
    def test_PreparePowerTwoState_inverse(self,bitsize):
        '''
        Tests the gate inverse correctly uncomputes by checking all qubits return to initial state when gate*(gate)**(-1) is applied.
        '''
        # initialilze operator
        gate = PreparePowerTwoState(bitsize=bitsize)
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

    @pytest.mark.parametrize("bitsize",[4,7])
    def test_PreparePowerTwoState_decomposes(self,bitsize):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = PreparePowerTwoState(bitsize=bitsize)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_PreparePowerTwoState_qasm(self):
        '''
        Tests qasm printing functionality.
        '''
        context = cirq.DecompositionContext(gam)
        bitsize = 5
        gate = PreparePowerTwoState(bitsize=bitsize)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None
        for line in qasm:
            pass

    def test_PreparePowerTwoState_resources(self):
        '''
        Tests resources are counted correctly. Assumes cirq default decomposition for controlled Hadamards.
        '''
        bitsize=5
        gate = PreparePowerTwoState(bitsize=bitsize)
        resources = estimate_resources(gate,profile=True)

        cliffs_per_CH = 31
        num_CH = bitsize-2

        assert resources['T'] == 4*num_CH
        assert resources['Clifford'] == 2*bitsize + cliffs_per_CH*num_CH
        assert resources['LogicalQubits'] == bitsize + 1
        assert resources['Rotations'] == 0
        assert gate.t_complexity() == gate.decompose_bloq().t_complexity()