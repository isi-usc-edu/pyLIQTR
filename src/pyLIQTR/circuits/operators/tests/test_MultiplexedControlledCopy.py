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
from pyLIQTR.circuits.operators.ControlledCopy import MultiplexedControlledCopy
from pyLIQTR.utils.resource_analysis import estimate_resources
from pyLIQTR.circuits.operators.tests.utils import ignore_bookkeeping

class TestMultiplexedControlledCopy:

    def test_MultiplexedControlledCopy(self):
        '''
        Tests the gate carries out the expected operation by comparing simulation output with the expected output for different input values.
        '''
        bitsize=3
        gate = MultiplexedControlledCopy(target_bitsize=bitsize)
        helper = GateHelper(gate=gate)
        # circuit
        circuit = helper.circuit
        # set input vals
        a_input = np.array([[1,1,1],[0,1,1],[1,0,0]])
        # check state is copied for given selection index
        for ind in range(3):
            sel_input = QUInt(2).to_bits(ind)
            input_state = [1]+sel_input+list(a_input.flatten()) + [0]*(bitsize)
            expected_out = [1]+sel_input+list(a_input.flatten())+list(a_input[ind])
            # assert equivalence of simulated circuit results and expected output
            result = cirq.Simulator(dtype=np.complex128).simulate(
            circuit, initial_state=input_state, qubit_order=[*helper.quregs['control'],*helper.quregs['selection'],*helper.quregs['original'].flatten(),*helper.quregs['target']])
            ## isolate string from inside ket
            actual = result.dirac_notation(decimals=2).split("|")[1][:-1]
            ## create expected string
            should_be = "".join(str(x) for x in expected_out)
            ## assert equivalence
            assert actual == should_be, f'\n{actual=}\n{should_be=}'

    def test_MultiplexedControlledCopy_inverse(self):
        '''
        Tests the gate inverse correctly uncomputes by checking all qubits return to initial state when gate*(gate)**(-1) is applied.
        '''
        # initialilze operator
        gate = MultiplexedControlledCopy(target_bitsize=3)
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

    @pytest.mark.parametrize("bitsize",[2,5])
    def test_MultiplexedControlledCopy_decomposes(self,bitsize):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = MultiplexedControlledCopy(target_bitsize=bitsize)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_MultiplexedControlledCopy_qasm(self):
        '''
        Tests qasm printing functionality.
        '''
        bitsize = 3
        gate = MultiplexedControlledCopy(target_bitsize=bitsize)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True)
        assert qasm is not None
        for line in qasm:
            pass

    def test_MultiplexedControlledCopy_resources(self):
        '''
        Tests resources are counted correctly.
        '''
        bitsize=3
        gate = MultiplexedControlledCopy(target_bitsize=bitsize)
        resources = estimate_resources(gate)

        assert resources['T'] == 3*4*bitsize + 4*2# different from reference by 2 Toffolis (for unary iteration)
        assert estimate_resources(gate.adjoint())['T'] == 4*2 # only 2 Toffolis from iteration
        assert resources['LogicalQubits'] == 4*bitsize+3
        assert gate.t_complexity() == gate.decompose_bloq().t_complexity()

    @pytest.mark.parametrize("bitsize",[2,5])
    def test_MultiplexedControlledCopy_callgraph(self,bitsize):
        '''
        Tests ability to generate call_graph and that call_graph counts match decomposition counts.
        '''
        gate = MultiplexedControlledCopy(target_bitsize=bitsize)
        call_graph, _ = gate.call_graph()

        call_graph_counts = gate.bloq_counts()
        cbloq = gate.decompose_bloq()
        decomp_counts = cbloq.bloq_counts(generalizer=ignore_bookkeeping)

        assert call_graph_counts == decomp_counts