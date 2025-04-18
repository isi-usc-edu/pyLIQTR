"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq

from qualtran.cirq_interop.testing import GateHelper
from pyLIQTR.utils.global_ancilla_manager import gam as gam
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.PrepareNuState import PrepareNuState
from pyLIQTR.utils.resource_analysis import estimate_resources
from pyLIQTR.utils.circuit_decomposition import decompose_once, generator_decompose

class TestPrepareNuState:    

    def test_PrepareNuState_decomposes(self):
        '''
        Tests gate decomposition existence.
        '''
        num_bits_p=3
        m_param=2**(2*num_bits_p+2+2) # reference assumes nm>2np+2
        # initialilze operator
        gate = PrepareNuState(num_bits_p=num_bits_p,m_param=m_param)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.Circuit(generator_decompose(operation))
        assert(helper.circuit != decomposed)

    def test_PrepareNuState_qasm(self):
        '''
        Tests qasm printing functionality.
        '''
        num_bits_p = 3
        m_param=2**(2*num_bits_p+2+2) # reference assumes nm>2np+2
        context = cirq.DecompositionContext(gam)
        gate = PrepareNuState(num_bits_p=num_bits_p,m_param=m_param)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None
        for line in qasm:
            pass

    @pytest.mark.parametrize("num_bits_p",[3,8])
    def test_PrepareNuState_resources(self,num_bits_p):
        '''
        Tests resources are consistent with reference and decomposition.
        '''
        n_m = 2*num_bits_p+2+2
        m_param=2**(n_m) # reference assumes nm>2np+2
        gate = PrepareNuState(num_bits_p=num_bits_p,m_param=m_param)
        helper = GateHelper(gate=gate)
        circuit = helper.circuit
        decomposed = decompose_once(circuit)
        resources = estimate_resources(gate,profile=True)
        resources_decomposed = estimate_resources(decomposed,profile=True)

        # differs from reference eq 90 by + 5? but all components match reference
        assert resources['T'] == 4*(3*num_bits_p**2+11*num_bits_p+2+4*n_m*(num_bits_p+1)) 
        assert resources['Rotations'] == 0

        assert resources == resources_decomposed

    @pytest.mark.parametrize("num_bits_p",[3,8])
    def test_PrepareNuState_callgraph(self,num_bits_p):
        '''
        Tests ability to generate call_graph
        '''
        m_param=2**(2*num_bits_p+2+2) # reference assumes nm>2np+2
        gate = PrepareNuState(num_bits_p=num_bits_p,m_param=m_param)
        call_graph, _ = gate.call_graph()