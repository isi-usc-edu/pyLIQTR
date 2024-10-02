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


class TestFermiHubbardSquareEncoding:

    @pytest.fixture(scope="class")
    def fh_instance(self):
        model = getInstance('FermiHubbard',shape=(8,8), J=-1.0, U=4.0,cell=SquareLattice)
        return model

    def test_FermiHubbardSquare_decomposes(self, fh_instance):
        '''
        Tests gate decomposition existence.
        '''
        encoding = getEncoding(instance=fh_instance, encoding=VALID_ENCODINGS.FermiHubbardSquare)
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

    def test_FermiHubbardSquare_qasm(self,fh_instance):
        '''
        Tests qasm printing functionality.
        '''
        # create registers
        encoding = getEncoding(instance=fh_instance, encoding=VALID_ENCODINGS.FermiHubbardSquare)
        num_qubits = cirq.num_qubits(encoding.circuit)
        qubits = cirq.LineQubit.range(num_qubits)
        # initialize operator
        operation = encoding.on(*qubits)

        qasm = openqasm(operation,rotation_allowed=True)
        assert qasm is not None
        for line in qasm:
            pass

    def test_FermiHubbardSquare_resources(self, fh_instance):
        '''
        Tests estimate_resources executes without error.
        '''
        encoding = getEncoding(instance=fh_instance, encoding=VALID_ENCODINGS.FermiHubbardSquare)
        resources = estimate_resources(encoding.circuit)