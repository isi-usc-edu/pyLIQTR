"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq
from pyLIQTR.ProblemInstances.getInstance import *
from pyLIQTR.BlockEncodings.getEncoding import *
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.utils.resource_analysis import estimate_resources
from pyLIQTR.clam.lattice_definitions import SquareLattice


class TestPauliLCUEncoding:

    @pytest.fixture(scope="class")
    def heis_instance(self):
        N=3

        J_x  =  J_y  =  0.5;              J_z = 1.0
        h_x  =  1.0;      h_y = 0.0;       h_z = 0.5

        model  =  getInstance("Heisenberg", 
                            shape=(N,N), 
                            J=(J_x,J_y,J_z), 
                            h=(h_x,h_y,h_z), 
                            cell=SquareLattice )
        return model

    def test_PauliLCU_decomposes(self, heis_instance):
        '''
        Tests gate decomposition existence.
        '''
        encoding = getEncoding(instance=heis_instance, encoding=VALID_ENCODINGS.PauliLCU)
        num_qubits = cirq.num_qubits(encoding.circuit)
        qubits = cirq.LineQubit.range(num_qubits)
        # initialize operator
        operation = encoding.on(*qubits)
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        assert([operation] != decomposed_once)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_PauliLCU_AS_decomposes(self, heis_instance):
        '''
        Tests gate decomposition existence.
        '''
        encoding = getEncoding(instance=heis_instance, encoding=VALID_ENCODINGS.PauliLCU, prepare_type='AS')
        num_qubits = cirq.num_qubits(encoding.circuit)
        qubits = cirq.LineQubit.range(num_qubits)
        # initialize operator
        operation = encoding.on(*qubits)
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        assert([operation] != decomposed_once)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_PauliLCU_qasm(self,heis_instance):
        '''
        Tests qasm printing functionality.
        '''
        # create registers
        encoding = getEncoding(instance=heis_instance, encoding=VALID_ENCODINGS.PauliLCU)
        num_qubits = cirq.num_qubits(encoding.circuit)
        qubits = cirq.LineQubit.range(num_qubits)
        # initialize operator
        operation = encoding.on(*qubits)

        qasm = openqasm(operation,rotation_allowed=True)
        assert qasm is not None
        for line in qasm:
            pass

    def test_PauliLCU_AS_qasm(self,heis_instance):
        '''
        Tests qasm printing functionality.
        '''
        # create registers
        encoding = getEncoding(instance=heis_instance, encoding=VALID_ENCODINGS.PauliLCU,prepare_type='AS')
        num_qubits = cirq.num_qubits(encoding.circuit)
        qubits = cirq.LineQubit.range(num_qubits)
        # initialize operator
        operation = encoding.on(*qubits)

        qasm = openqasm(operation,rotation_allowed=True)
        assert qasm is not None
        for line in qasm:
            pass

    def test_PauliLCU_resources(self, heis_instance):
        '''
        Tests estimate_resources executes without error.
        '''
        encoding = getEncoding(instance=heis_instance, encoding=VALID_ENCODINGS.PauliLCU)
        resources = estimate_resources(encoding.circuit)

    def test_PauliLCU_AS_resources(self, heis_instance):
        '''
        Tests estimate_resources executes without error.
        '''
        encoding = getEncoding(instance=heis_instance, encoding=VALID_ENCODINGS.PauliLCU, prepare_type='AS')
        resources = estimate_resources(encoding.circuit)