"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq
import numpy as np

from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim, GateHelper
from pyLIQTR.utils.global_ancilla_manager import gam as gam
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.UniformSuperpositionIJ import UniformSuperpositionIJFirstQuantization
from pyLIQTR.utils.resource_analysis import estimate_resources

class TestUniformSuperpositionIJ:    
                
    def test_UniformSuperpositionIJ_inverse(self):
        '''
        Tests the gate inverse correctly uncomputes by checking all qubits return to initial state when gate*(gate)**(-1) is applied.
        '''
        # initialilze operator
        gate = UniformSuperpositionIJFirstQuantization(eta=3)
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

    def test_UniformSuperpositionIJ_decomposes(self):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = UniformSuperpositionIJFirstQuantization(eta=3)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_UniformSuperpositionIJ_qasm(self):
        '''
        Tests qasm printing functionality.
        '''
        context = cirq.DecompositionContext(gam)
        bitsize = 3
        gate = UniformSuperpositionIJFirstQuantization(eta=3)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None
        for line in qasm:
            pass

    def test_UniformSuperpositionIJ_resources(self):
        '''
        Tests resources are consistent with decomposition.
        '''
        eta=3
        gate = UniformSuperpositionIJFirstQuantization(eta=eta)

        assert gate.t_complexity() == gate.decompose_bloq().t_complexity()