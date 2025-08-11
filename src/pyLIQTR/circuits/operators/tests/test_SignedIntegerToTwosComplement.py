"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq
import numpy as np

from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim
from qualtran.cirq_interop.testing import GateHelper
from qualtran.resource_counting.generalizers import ignore_split_join
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.SignedIntegerToTwosComplement import SignedIntegerToTwosComplement
from pyLIQTR.utils.resource_analysis import estimate_resources

class TestSignedIntegerToTwosComplement:

    @pytest.mark.parametrize("bitsize,initial_state,expected_out",[(4,[0,1,1,1],[0,1,1,1]),(4,[1,0,1,0],[1,1,1,0]),(5,[0,1,0,1,0],[0,1,0,1,0]),(5,[1,0,1,0,0],[1,1,1,0,0])])
    def test_SignedIntegerToTwosComplement(self,bitsize,initial_state,expected_out):
        '''
        Tests the gate carries out the expected operation by comparing simulation output with the expected output for different input values.
        '''
        gate = SignedIntegerToTwosComplement(bitsize=bitsize)
        helper = GateHelper(gate=gate)
        # operation = helper.operation
        # circuit
        qubit_order=helper.quregs['x']
        circuit = helper.circuit
        assert_circuit_inp_out_cirqsim(circuit,inputs=initial_state, outputs=expected_out,qubit_order=qubit_order)

    @pytest.mark.parametrize("bitsize,initial_state,expected_out",[(4,[0,1,1,1],[0,1,1,1]),(4,[1,1,1,0],[1,0,1,0]),(5,[0,1,0,1,0],[0,1,0,1,0]),(5,[1,1,1,0,0],[1,0,1,0,0])])
    def test_SignedIntegerToTwosComplement_reverse(self,bitsize,initial_state,expected_out):
        '''
        Tests the gate inverse carries out the expected operation by comparing simulation output with the expected output for different input values.
        '''
        gate = SignedIntegerToTwosComplement(bitsize=bitsize).adjoint()
        helper = GateHelper(gate=gate)
        # operation = helper.operation
        # circuit
        qubit_order=helper.quregs['x']
        circuit = helper.circuit
        assert_circuit_inp_out_cirqsim(circuit,inputs=initial_state, outputs=expected_out,qubit_order=qubit_order)
        

    @pytest.mark.parametrize("bitsize",[4,7])
    def test_SignedIntegerToTwosComplement_inverse(self,bitsize):
        '''
        Tests the gate inverse correctly uncomputes by checking all qubits return to initial state when gate*(gate)**(-1) is applied.
        '''
        # initialilze operator
        gate = SignedIntegerToTwosComplement(bitsize=bitsize)
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
    def test_SignedIntegerToTwosComplement_decomposes(self,bitsize):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = SignedIntegerToTwosComplement(bitsize=bitsize)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_SignedIntegerToTwosComplement_qasm(self):
        '''
        Tests qasm printing functionality.
        '''
        bitsize = 5
        gate = SignedIntegerToTwosComplement(bitsize=bitsize)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True)
        assert qasm is not None
        for line in qasm:
            pass

    def test_SignedIntegerToTwosComplement_resources(self):
        '''
        Tests resources are counted correctly. Assumes cirq default decomposition for controlled Hadamards.
        '''
        bitsize=5
        gate = SignedIntegerToTwosComplement(bitsize=bitsize)
        resources = estimate_resources(gate,profile=True)

        cliffs_for_and = 9
        cliffs_for_and_adj = 4
        num_ands = bitsize-2
        cnots_from_add = (bitsize-1 - 2) * 6 + 3

        assert resources['T'] == 4*num_ands
        assert resources['Clifford'] == bitsize-1 + num_ands*(cliffs_for_and+cliffs_for_and_adj) + cnots_from_add
        assert resources['LogicalQubits'] == bitsize
        assert resources['Rotations'] == 0
        assert gate.t_complexity() == gate.decompose_bloq().t_complexity()

    @pytest.mark.parametrize("bitsize",[2,6])
    def test_SignedIntegerToTwosComplement_callgraph(self,bitsize):
        '''
        Tests ability to generate call_graph and that call_graph counts match decomposition counts.
        '''
        gate = SignedIntegerToTwosComplement(bitsize=bitsize)
        call_graph, _ = gate.call_graph()

        call_graph_counts = gate.bloq_counts()
        cbloq = gate.decompose_bloq()
        decomp_counts = cbloq.bloq_counts(generalizer=ignore_split_join)

        assert call_graph_counts == decomp_counts