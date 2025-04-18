"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq

from qualtran.cirq_interop.testing import GateHelper
from pyLIQTR.utils.global_ancilla_manager import gam as gam
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.NuInequalityTest import NuInequalityTest
from pyLIQTR.utils.resource_analysis import estimate_resources
from pyLIQTR.utils.circuit_decomposition import decompose_once, generator_decompose

class TestNuInequalityTest:    

    @pytest.mark.parametrize("num_bits_p",[3,8])
    def test_NuInequalityTest_decomposes(self,num_bits_p):
        '''
        Tests gate decomposition existence.
        '''
        num_bits_m=2*num_bits_p+2+2 # reference assumes nm>2np+2
        # initialilze operator
        gate = NuInequalityTest(num_bits_p=num_bits_p,num_bits_m=num_bits_m)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.Circuit(generator_decompose(operation))
        assert(helper.circuit != decomposed)

    def test_NuInequalityTest_qasm(self):
        '''
        Tests qasm printing functionality.
        '''
        num_bits_p = 8
        num_bits_m=2*num_bits_p+2+2 # reference assumes nm>2np+2
        context = cirq.DecompositionContext(gam)
        gate = NuInequalityTest(num_bits_p=num_bits_p,num_bits_m=num_bits_m)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None
        for line in qasm:
            pass

    @pytest.mark.parametrize("num_bits_p",[3,8])
    def test_NuInequalityTest_resources(self,num_bits_p):
        '''
        Tests resources are consistent with reference and decomposition.
        '''
        num_bits_m=2*num_bits_p+2+2 # reference assumes nm>2np+2
        gate = NuInequalityTest(num_bits_p=num_bits_p,num_bits_m=num_bits_m)
        helper = GateHelper(gate=gate)
        circuit = helper.circuit
        decomposed = decompose_once(circuit)
        decomp_alloc_qubits = num_bits_p+num_bits_m+2
        resources = estimate_resources(gate,profile=True)
        resources['LogicalQubits'] += decomp_alloc_qubits - 3 #-3 bc ignore nu sign bits
        resources_decomposed = estimate_resources(decomposed,profile=True)

        # matches reference
        assert resources['T'] == 4*(3*num_bits_p**2+num_bits_p+1+4*num_bits_m*(num_bits_p+1)) 
        assert resources['Rotations'] == 0

        assert resources == resources_decomposed

    @pytest.mark.parametrize("num_bits_p",[3,8])
    def test_NuInequalityTest_callgraph(self,num_bits_p):
        '''
        Tests ability to generate call_graph
        '''
        num_bits_m=2*num_bits_p+2+2 # reference assumes nm>2np+2
        gate = NuInequalityTest(num_bits_p=num_bits_p,num_bits_m=num_bits_m)
        call_graph, _ = gate.call_graph()