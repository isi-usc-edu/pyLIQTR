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
from pyLIQTR.utils.global_ancilla_manager import gam as gam
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.FlaggedPrepareUniformSuperposition import FlaggedPrepareUniformSuperposition
from pyLIQTR.utils.resource_analysis import estimate_resources

class TestFlaggedPrepareUniformSuperposition:

    @pytest.mark.parametrize("num_states",[4,6])
    def test_FlaggedPrepareUniformSuperposition(self,num_states):
        '''
        Tests the gate carries out the expected operation by comparing simulation output with the expected output for different input values.
        '''
        gate = FlaggedPrepareUniformSuperposition(d=num_states)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # circuit
        circuit = cirq.Circuit(operation)
        result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=[*helper.quregs['target'],*helper.quregs['success'],*helper.quregs['less_than_ancilla'],*helper.quregs['rot_ancilla']])
        # isolate successful state on target bits
        output_state = cirq.partial_trace_of_state_vector_as_mixture(result.final_state_vector,keep_indices=[*range(gate.nd_bits)])[-1][1]
        # set up expected output state
        expected_out = np.array([1/np.sqrt(num_states)+0j if i<num_states else 0j for i in range(2**gate.nd_bits)])
        # expected_out
        cirq.testing.assert_allclose_up_to_global_phase(expected_out, output_state, atol=1e-6)

        # check less than ancilla returns to zero state
        less_than_output = cirq.sub_state_vector(result.final_state_vector,keep_indices=[gate.nd_bits+1])
        cirq.testing.assert_allclose_up_to_global_phase(less_than_output,np.array([1+0j,0+0j]),atol=1e-10)
        

    @pytest.mark.parametrize("num_states",[4,6])
    def test_FlaggedPrepareUniformSuperposition_inverse(self,num_states):
        '''
        Tests the gate inverse correctly uncomputes by checking all qubits return to initial state when gate*(gate)**(-1) is applied.
        '''
        # initialilze operator
        gate = FlaggedPrepareUniformSuperposition(d=num_states)
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

    @pytest.mark.parametrize("num_states",[2,6])
    def test_FlaggedPrepareUniformSuperposition_decomposes(self,num_states):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = FlaggedPrepareUniformSuperposition(d=num_states)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_FlaggedPrepareUniformSuperposition_qasm(self):
        '''
        Tests qasm printing functionality.
        '''
        context = cirq.DecompositionContext(gam)
        num_states = 6
        gate = FlaggedPrepareUniformSuperposition(d=num_states)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None
        for line in qasm:
            pass

    @pytest.mark.parametrize("num_states",[3,6,9])
    def test_FlaggedPrepareUniformSuperposition_resources(self,num_states):
        '''
        Tests resources are counted correctly. Assumes cirq default decomposition for controlled Hadamards.
        '''
        gate = FlaggedPrepareUniformSuperposition(d=num_states)
        resources = estimate_resources(gate,profile=True)

        d, eta = num_states, 0
        while d > 1 and d % 2 == 0:
            eta += 1
            d = d // 2
        l, logL = int(d), gate.nd_bits - eta

        num_H = gate.nd_bits + 1 + 2*(logL + 1)
        num_X = 4

        lbl = l.bit_length()
        num_ones = l.bit_count()
        num_zeros = lbl - num_ones
        num_less_than = 4
        t_per_less_than = 4*logL
        cliffs_per_less_than = (9+4)*logL + 2*(2*num_ones + num_zeros+ 1) +2

        t_for_mcp = 4*(logL-1)
        cliff_for_mcp = (9+4)*(logL-1) + 4*(logL) + 1

        assert resources['T'] == num_less_than*t_per_less_than + t_for_mcp
        assert resources['Clifford'] == num_less_than*cliffs_per_less_than + num_H + num_X + cliff_for_mcp + 2
        assert resources['LogicalQubits'] == gate.nd_bits + 3
        assert resources['Rotations'] == 2
        assert gate.t_complexity() == gate.decompose_bloq().t_complexity()

    @pytest.mark.parametrize("num_states",[2,4])
    def test_FlaggedPrepareUniformSuperposition_resources_pow2(self,num_states):
        '''
        Tests resources are counted correctly. Assumes cirq default decomposition for controlled Hadamards.
        '''
        gate = FlaggedPrepareUniformSuperposition(d=num_states)
        resources = estimate_resources(gate,profile=True)

        assert resources['T'] == 0
        assert resources['Clifford'] == gate.nd_bits + 1
        assert resources['LogicalQubits'] == gate.nd_bits + 3
        assert resources['Rotations'] == 0
        assert gate.t_complexity() == gate.decompose_bloq().t_complexity()

    @pytest.mark.parametrize("num_states",[2,6])
    def test_FlaggedPrepareUniformSuperposition_callgraph(self,num_states):
        '''
        Tests ability to generate call_graph and that call_graph counts match decomposition counts.
        '''
        gate = FlaggedPrepareUniformSuperposition(d=num_states)
        call_graph, _ = gate.call_graph()

        call_graph_counts = gate.bloq_counts()
        cbloq = gate.decompose_bloq()
        decomp_counts = cbloq.bloq_counts(generalizer=ignore_split_join)

        assert call_graph_counts == decomp_counts