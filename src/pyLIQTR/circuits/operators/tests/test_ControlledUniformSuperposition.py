"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq
import numpy as np

from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim
from qualtran.cirq_interop.testing import GateHelper
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.ControlledUniformSuperposition import ControlledPrepareUniformSuperposition
from pyLIQTR.utils.resource_analysis import estimate_resources

class TestControlledUniformSuperposition:

    def test_ControlledUniformSuperposition_inverse(self):
        '''
        Tests the gate inverse correctly uncomputes by checking all qubits return to initial state when gate*(gate)**(-1) is applied.
        '''
        br = 3
        bphi = 3
        Xi = 2 # the number of states is Xi+1 since indexing starts at 0
        n_Xi = (Xi).bit_length()
        # initialilze operator
        gate = ControlledPrepareUniformSuperposition(br,bphi,n_Xi)
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

    def test_ControlledUniformSuperposition_decomposes(self):
        '''
        Tests gate decomposition existence.
        '''
        br = 3
        bphi = 3
        Xi = 2 # the number of states is Xi+1 since indexing starts at 0
        n_Xi = (Xi).bit_length()
        # initialilze operator
        gate = ControlledPrepareUniformSuperposition(br,bphi,n_Xi)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_ControlledUniformSuperposition_qasm(self):
        '''
        Tests qasm printing functionality.
        '''
        br = 3
        bphi = 3
        Xi = 2 # the number of states is Xi+1 since indexing starts at 0
        n_Xi = (Xi).bit_length()
        # initialilze operator
        gate = ControlledPrepareUniformSuperposition(br,bphi,n_Xi)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True)
        assert qasm is not None
        for line in qasm:
            pass

    def test_ControlledUniformSuperposition_resources(self):
        '''
        Tests resources are counted correctly.
        '''
        br = 3
        bphi = 3
        Xi = 2 # the number of states is Xi+1 since indexing starts at 0
        n_Xi = (Xi).bit_length()
        # initialilze operator
        gate = ControlledPrepareUniformSuperposition(br,bphi,n_Xi)
        resources = estimate_resources(gate)

        assert gate.t_complexity() == gate.decompose_bloq().t_complexity()