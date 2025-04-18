"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq
import numpy as np

from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim, GateHelper
from qualtran import QUInt
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.Product import Product
from pyLIQTR.utils.resource_analysis import estimate_resources
from pyLIQTR.circuits.operators.tests.utils import ignore_bookkeeping

class TestProduct:

    @pytest.mark.parametrize("a_bitsize, b_bitsize",[(2,2),(4,3)])
    def test_Product(self,a_bitsize,b_bitsize):
        '''
        Tests the gate carries out the expected operation by comparing simulation output with the expected output for different input values.
        '''
        gate = Product(a_bitsize=a_bitsize,b_bitsize=b_bitsize)
        helper = GateHelper(gate=gate)
        # circuit
        circuit = helper.circuit
        # set input vals to max possible
        a_val = 2**a_bitsize-1
        b_val = 2**b_bitsize-1
        a_input = QUInt(a_bitsize).to_bits(a_val)
        b_input = QUInt(b_bitsize).to_bits(b_val)
        input_state = a_input + b_input + [0]*(a_bitsize+b_bitsize)
        # expected output is product of a and b
        product_state = QUInt(a_bitsize+b_bitsize).to_bits(a_val*b_val)
        expected_out = a_input + b_input + product_state
        # assert equivalence of simulated circuit results and expected output
        assert_circuit_inp_out_cirqsim(circuit=circuit,qubit_order=[*helper.quregs['A'],*helper.quregs['B'],*helper.quregs['target']],inputs=input_state,outputs=expected_out)

    def test_Product_ValueError(self):
        '''
        Tests the gate raises ValueError for b_bitsize > a_bitsize.
        '''
        with pytest.raises(ValueError, match='must be less than or equal to a_bitsize'):
            gate = Product(a_bitsize=2,b_bitsize=3)


    @pytest.mark.parametrize("bitsize",[2])
    def test_Product_inverse(self,bitsize):
        '''
        Tests the gate inverse correctly uncomputes by checking all qubits return to initial state when gate*(gate)**(-1) is applied.
        '''
        # initialilze operator
        gate = Product(a_bitsize=bitsize)
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

    @pytest.mark.parametrize("bitsize",[3])
    def test_Product_decomposes(self,bitsize):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = Product(a_bitsize=bitsize)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_Product_qasm(self):
        '''
        Tests qasm printing functionality.
        '''
        bitsize = 3
        gate = Product(a_bitsize=bitsize)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True)
        assert qasm is not None
        for line in qasm:
            pass

    def test_Product_resources(self):
        '''
        Tests resources are counted correctly.
        '''
        bitsize=3
        gate = Product(a_bitsize=bitsize)
        resources = estimate_resources(gate)
        
        assert resources['T'] == 4*(2*bitsize**2-bitsize) # matches reference
        assert resources['LogicalQubits'] == 4*bitsize # doesn't count ancilla
        assert gate.t_complexity() == gate.decompose_bloq().t_complexity()

    @pytest.mark.parametrize("bitsize",[2,6])
    def test_Product_callgraph(self,bitsize):
        '''
        Tests ability to generate call_graph and that call_graph counts match decomposition counts.
        '''
        gate = Product(a_bitsize=bitsize)
        call_graph, _ = gate.call_graph()

        call_graph_counts = gate.bloq_counts()
        cbloq = gate.decompose_bloq()
        decomp_counts = cbloq.bloq_counts(generalizer=ignore_bookkeeping)

        assert call_graph_counts == decomp_counts