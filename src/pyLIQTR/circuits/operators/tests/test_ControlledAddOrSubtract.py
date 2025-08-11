"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq

from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim, GateHelper
from qualtran import QUInt
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.ControlledAddOrSubtract import ControlledAddOrSubtract
from pyLIQTR.utils.resource_analysis import estimate_resources
from pyLIQTR.utils.circuit_decomposition import generator_decompose

class TestControlledAddOrSubtract:

    @pytest.mark.parametrize("ctrl, a_bitsize, b_bitsize, a_val, b_val",[(1,2,2,1,2),(0,2,3,1,5)])
    def test_ControlledAddOrSubtract(self,ctrl,a_bitsize,b_bitsize,a_val,b_val):
        '''
        Tests the gate carries out the expected operation by comparing simulation output with the expected output for different input values.
        '''
        a_dtype = QUInt(a_bitsize)
        b_dtype = QUInt(b_bitsize)
        gate = ControlledAddOrSubtract(a_dtype=a_dtype,b_dtype=b_dtype)
        helper = GateHelper(gate=gate)
        # circuit
        circuit = helper.circuit
        # setup input state
        a_input = a_dtype.to_bits(a_val)
        b_input = b_dtype.to_bits(b_val)
        input_state = [ctrl] + a_input + b_input
        # expected output is sum (ctrl=1) / difference (ctrl=0) of a and b
        expected_out = [ctrl] + a_input + b_dtype.to_bits(b_val-(-1)**ctrl*a_val)
        # assert equivalence of simulated circuit results and expected output
        assert_circuit_inp_out_cirqsim(circuit=circuit,qubit_order=[*helper.quregs['ctrl'],*helper.quregs['a'],*helper.quregs['b']],inputs=input_state,outputs=expected_out)


    @pytest.mark.parametrize("ctrl, a_bitsize, b_bitsize, a_val, b_val",[(1,2,2,1,2),(0,2,3,1,5)])
    def test_ControlledAddOrSubtract_inverse(self,ctrl,a_bitsize,b_bitsize,a_val,b_val):
        '''
        Tests the gate inverse correctly uncomputes by checking all qubits return to initial state when gate*(gate)**(-1) is applied.
        '''
        # initialilze operator
        a_dtype = QUInt(a_bitsize)
        b_dtype = QUInt(b_bitsize)
        gate = ControlledAddOrSubtract(a_dtype=a_dtype,b_dtype=b_dtype)
        helper = GateHelper(gate=gate)
        # circuit
        circuit = helper.circuit
        circuit.append(helper.operation**(-1))
        # setup input state
        a_input = a_dtype.to_bits(a_val)
        b_input = b_dtype.to_bits(b_val)
        input_state = [ctrl] + a_input + b_input
        # output state equals input
        # verify simulated results
        assert_circuit_inp_out_cirqsim(circuit,qubit_order=[*helper.quregs['ctrl'],*helper.quregs['a'],*helper.quregs['b']],inputs=input_state,outputs=input_state)

    @pytest.mark.parametrize("a_bitsize, b_bitsize",[(2,2),(2,3)])
    def test_ControlledAddOrSubtract_decomposes(self, a_bitsize, b_bitsize):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        a_dtype = QUInt(a_bitsize)
        b_dtype = QUInt(b_bitsize)
        gate = ControlledAddOrSubtract(a_dtype=a_dtype,b_dtype=b_dtype)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = generator_decompose(operation)
        assert([operation] != decomposed)

    @pytest.mark.parametrize("a_bitsize, b_bitsize",[(2,2),(2,3)])
    def test_ControlledAddOrSubtract_qasm(self,a_bitsize,b_bitsize):
        '''
        Tests qasm printing functionality.
        '''
        a_dtype = QUInt(a_bitsize)
        b_dtype = QUInt(b_bitsize)
        gate = ControlledAddOrSubtract(a_dtype=a_dtype,b_dtype=b_dtype)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True)
        assert qasm is not None
        for line in qasm:
            pass

    @pytest.mark.parametrize("a_bitsize, b_bitsize",[(2,2),(2,3)])
    def test_ControlledAddOrSubtract_resources(self,a_bitsize, b_bitsize):
        '''
        Tests resources are counted correctly.
        '''
        a_dtype = QUInt(a_bitsize)
        b_dtype = QUInt(b_bitsize)
        gate = ControlledAddOrSubtract(a_dtype=a_dtype,b_dtype=b_dtype)
        resources = estimate_resources(gate)
        
        assert resources['T'] == 4*(b_bitsize-1) # only t gates come from cost of addition
        assert gate.t_complexity() == gate.decompose_bloq().t_complexity()

    @pytest.mark.parametrize("a_bitsize, b_bitsize",[(2,2),(2,3)])
    def test_ControlledAddOrSubtract_callgraph(self,a_bitsize,b_bitsize):
        '''
        Tests ability to generate call_graph
        '''
        a_dtype = QUInt(a_bitsize)
        b_dtype = QUInt(b_bitsize)
        gate = ControlledAddOrSubtract(a_dtype=a_dtype,b_dtype=b_dtype)
        call_graph, _ = gate.call_graph()