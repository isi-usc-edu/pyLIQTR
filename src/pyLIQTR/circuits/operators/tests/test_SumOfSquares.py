"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq

from qualtran.cirq_interop.testing import GateHelper
from pyLIQTR.utils.global_ancilla_manager import gam as gam
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.SumOfSquares import SumOf3Squares
from pyLIQTR.utils.resource_analysis import estimate_resources
from pyLIQTR.utils.circuit_decomposition import decompose_once, generator_decompose

class TestSumOf3Squares:    

    @pytest.mark.parametrize("bitsize",[3,4,8,9])
    def test_SumOf3Squares_decomposes(self,bitsize):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = SumOf3Squares(num_bits_p=bitsize)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = generator_decompose(operation)
        assert([operation] != decomposed)

    @pytest.mark.parametrize("bitsize",[3,8])
    def test_SumOf3Squares_qasm(self,bitsize):
        '''
        Tests qasm printing functionality.
        '''
        context = cirq.DecompositionContext(gam)
        gate = SumOf3Squares(num_bits_p=bitsize)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None
        for line in qasm:
            pass

    @pytest.mark.parametrize("bitsize",[3,4,8,9])
    def test_SumOf3Squares_resources(self,bitsize):
        '''
        Tests resources are consistent with reference and decomposition.
        '''
        gate = SumOf3Squares(num_bits_p=bitsize)
        helper = GateHelper(gate=gate)
        circuit = helper.circuit
        decomposed = decompose_once(circuit)
        resources = estimate_resources(gate,profile=True)
        resources_decomposed = estimate_resources(decomposed,profile=True)

        # matches reference
        assert resources['T'] == 4*(3*bitsize**2 - bitsize - 1) 
        assert resources['Rotations'] == 0
        assert estimate_resources(gate.adjoint())['T'] == 0

        assert resources == resources_decomposed

    @pytest.mark.parametrize("bitsize",[3,4,8,9])
    def test_SumOf3Squares_callgraph(self,bitsize):
        '''
        Tests ability to generate call_graph
        '''
        gate = SumOf3Squares(num_bits_p=bitsize)
        call_graph, _ = gate.call_graph()