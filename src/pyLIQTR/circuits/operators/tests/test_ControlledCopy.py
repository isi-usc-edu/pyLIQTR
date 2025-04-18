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
from pyLIQTR.circuits.operators.ControlledCopy import ControlledCopy
from pyLIQTR.utils.resource_analysis import estimate_resources
from pyLIQTR.circuits.operators.tests.utils import ignore_bookkeeping

class TestControlledCopy:

    @pytest.mark.parametrize("bitsize, control_val",[(2,1),(3,0)])
    def test_ControlledCopy(self,bitsize, control_val):
        '''
        Tests the gate carries out the expected operation by comparing simulation output with the expected output for different input values.
        '''
        gate = ControlledCopy(bitsize=bitsize,control_val=control_val)
        helper = GateHelper(gate=gate)
        # circuit
        circuit = helper.circuit
        # set input vals to max possible
        a_input = QUInt(bitsize).to_bits(2**bitsize-1)
        # check state is copied for given control_val and not otherwise
        for cv_input in [control_val, abs(control_val-1)]:
            input_state = [cv_input]+a_input + [0]*(bitsize)
            if cv_input==control_val:
                expected_out = [cv_input]+a_input+a_input
            else:
                expected_out = input_state
            # assert equivalence of simulated circuit results and expected output
            assert_circuit_inp_out_cirqsim(circuit=circuit,qubit_order=[*helper.quregs['control'],*helper.quregs['original'],*helper.quregs['target']],inputs=input_state,outputs=expected_out)

    @pytest.mark.parametrize("bitsize, control_val",[(2,1),(2,0)])
    def test_ControlledCopy_inverse(self,bitsize, control_val):
        '''
        Tests the gate inverse correctly uncomputes by checking all qubits return to initial state when gate*(gate)**(-1) is applied.
        '''
        # initialilze operator
        gate = ControlledCopy(bitsize=bitsize,control_val=control_val)
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

    @pytest.mark.parametrize("bitsize, control_val",[(3,1),(3,0)])
    def test_ControlledCopy_decomposes(self,bitsize, control_val):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = ControlledCopy(bitsize=bitsize,control_val=control_val)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    @pytest.mark.parametrize("control_val",[1,0])
    def test_ControlledCopy_qasm(self,control_val):
        '''
        Tests qasm printing functionality.
        '''
        bitsize = 3
        gate = ControlledCopy(bitsize=bitsize, control_val=control_val)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True)
        assert qasm is not None
        for line in qasm:
            pass

    @pytest.mark.parametrize("control_val",[1,0])
    def test_ControlledCopy_resources(self, control_val):
        '''
        Tests resources are counted correctly.
        '''
        bitsize=3
        gate = ControlledCopy(bitsize=bitsize,control_val=control_val)
        resources = estimate_resources(gate)

        cliffs_for_and = 9
        if control_val:
            cliffs_for_zero_controls = 0
        else:
            cliffs_for_zero_controls = 2

        assert resources['T'] == 4*bitsize # matches reference
        assert resources['Clifford'] == cliffs_for_zero_controls + cliffs_for_and*bitsize
        assert resources['LogicalQubits'] ==2*bitsize+1
        assert estimate_resources(gate.adjoint())['T'] == 0
        assert gate.t_complexity() == gate.decompose_bloq().t_complexity()

    @pytest.mark.parametrize("bitsize, control_val",[(2,1),(6,0)])
    def test_ControlledCopy_callgraph(self,bitsize, control_val):
        '''
        Tests ability to generate call_graph and that call_graph counts match decomposition counts.
        '''
        gate = ControlledCopy(bitsize=bitsize,control_val=control_val)
        call_graph, _ = gate.call_graph()

        call_graph_counts = gate.bloq_counts()
        cbloq = gate.decompose_bloq()
        decomp_counts = cbloq.bloq_counts(generalizer=ignore_bookkeeping)

        assert call_graph_counts == decomp_counts