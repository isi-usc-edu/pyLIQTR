"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq

from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim, GateHelper
from pyLIQTR.utils.global_ancilla_manager import gam as gam
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.SelectT_FirstQuantized import SelectT_FirstQuantized
from pyLIQTR.utils.resource_analysis import estimate_resources
from pyLIQTR.utils.circuit_decomposition import decompose_once, generator_decompose

class TestSelectT_FirstQuantized:    

    def test_SelectT_FirstQuantized_decomposes(self):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = SelectT_FirstQuantized(num_bits_p=3)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = generator_decompose(operation)
        assert([operation] != decomposed)

    def test_SelectT_FirstQuantized_qasm(self):
        '''
        Tests qasm printing functionality.
        '''
        context = cirq.DecompositionContext(gam)
        bitsize = 3
        gate = SelectT_FirstQuantized(num_bits_p=3)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None
        for line in qasm:
            pass

    @pytest.mark.parametrize("bitsize",[3,8])
    def test_SelectT_FirstQuantized_resources(self,bitsize):
        '''
        Tests resources are consistent with decomposition.
        '''
        gate = SelectT_FirstQuantized(num_bits_p=bitsize)
        helper = GateHelper(gate=gate)
        circuit = helper.circuit
        decomposed = decompose_once(circuit)
        resources = estimate_resources(gate,profile=True)
        resources_decomposed = estimate_resources(decomposed,profile=True)
        decomp_ancilla_allocated = bitsize+2
        resources['LogicalQubits'] += decomp_ancilla_allocated

        mccp = 3*bitsize + 1
        mccp_adjoint = 2
        overall_control = 1

        # differs from ref by +2 for mccp adjoint
        # np vs np-1 discrepency
        # extra Toffoli for controlling copy out of w (step 1)
        assert resources['T'] == 4*(mccp+overall_control+mccp_adjoint + 2*bitsize + 1) 

        assert resources == resources_decomposed

    def test_SelectT_FirstQuantized_callgraph(self):
        '''
        Tests ability to generate call_graph.
        '''
        gate = SelectT_FirstQuantized(num_bits_p=3)
        call_graph, _ = gate.call_graph()