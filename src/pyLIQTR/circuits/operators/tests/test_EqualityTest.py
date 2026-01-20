"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq
import numpy as np

from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim, GateHelper
from pyLIQTR.utils.global_ancilla_manager import gam as gam
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.EqualityTest import EqualityTest
from pyLIQTR.utils.resource_analysis import estimate_resources

class TestEqualityTest:

    @pytest.mark.parametrize("bitsize",[2,3])
    def test_EqualityTest(self,bitsize):
        '''
        Tests the gate carries out the expected operation by comparing simulation output with the expected output for different input values.
        '''
        gate = EqualityTest(bitsize=bitsize)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # circuit
        circuit = cirq.Circuit(operation)
        # check simulation results. target qubit should end in 1 state only if i==j, other qubits should be unchanged.
        for i in range(2**bitsize):
            for j in range(2**bitsize):
                input_state = list(iter_bits(i,width=bitsize))+list(iter_bits(j,width=bitsize))+[0]
                output_state = input_state.copy()
                if i==j:
                    output_state[-1] = 1
                assert_circuit_inp_out_cirqsim(circuit,qubit_order=[*helper.quregs['A'],*helper.quregs['B'],*helper.quregs['target']],inputs=input_state,outputs=output_state)

    def test_EqualityTest_ValueError(self):
        '''
        Tests the gate raises ValueError for bitsize < 2.
        '''
        with pytest.raises(ValueError, match='bitsize must be greater than 2'):
            gate = EqualityTest(bitsize=1)  

    @pytest.mark.parametrize("bitsize",[4,7])
    def test_EqualityTest_inverse(self,bitsize):
        '''
        Tests the gate inverse correctly uncomputes by checking all qubits return to initial state when gate*(gate)**(-1) is applied.
        '''
        # initialilze operator
        gate = EqualityTest(bitsize=bitsize)
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
    def test_EqualityTest_decomposes(self,bitsize):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = EqualityTest(bitsize=bitsize)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_EqualityTest_qasm(self):
        '''
        Tests qasm printing functionality.
        '''
        context = cirq.DecompositionContext(gam)
        bitsize = 5
        gate = EqualityTest(bitsize=bitsize)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None
        for line in qasm:
            pass

    def test_EqualityTest_resources(self):
        '''
        Tests resources are counted correctly.
        '''
        bitsize=5
        gate = EqualityTest(bitsize=bitsize)
        resources = estimate_resources(gate)

        cliffs_for_zero_controls = 4*bitsize
        cliffs_for_and = 9
        cliffs_for_and_adj = 4
        num_CNOTS = 2*bitsize+1
        num_ands = bitsize-1

        assert resources['T'] == 4*(bitsize-1)
        assert resources['Clifford'] == num_CNOTS + num_ands*(cliffs_for_and+cliffs_for_and_adj) + cliffs_for_zero_controls
        assert resources['LogicalQubits'] == 2*bitsize + 1
        assert gate.t_complexity() == gate.decompose_bloq().t_complexity()