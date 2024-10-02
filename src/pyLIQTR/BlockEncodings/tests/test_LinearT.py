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

@pytest.fixture(scope="module")
def script_loc(request):
    '''Return the directory of the currently running test script'''

    return request.path.parent

class TestLinearTEncoding:

    @pytest.fixture(scope="class")
    def es_instance(self, script_loc):
        example_ham_filename = script_loc.joinpath('../../../../data/hamiltonians/mg_slab-4x4x2/magnesium-4x4x2-Ecut_0.5_scale_1.0.ham.hdf5')
        example_grid_filename = script_loc.joinpath('../../../../data/hamiltonians/mg_slab-4x4x2/magnesium-4x4x2-Ecut_0.5_scale_1.0.grid.hdf5')
        model = getInstance('ElectronicStructure',filenameH=example_ham_filename,filenameG=example_grid_filename)
        print(model)
        return model

    @pytest.fixture(scope="class")
    def lt_encoding(self, es_instance):
        return getEncoding(VALID_ENCODINGS.LinearT, instance=es_instance, energy_error=0.5, control_val=1)

    def test_LinearT_decomposes(self, lt_encoding):
        '''
        Tests gate decomposition existence.
        '''
        num_qubits = cirq.num_qubits(lt_encoding.circuit)
        qubits = cirq.LineQubit.range(num_qubits)
        # initialize operator
        operation = lt_encoding.on(*qubits)
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        assert([operation] != decomposed_once)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_LinearT_qasm(self,lt_encoding):
        '''
        Tests qasm printing functionality.
        '''
        # create registers
        num_qubits = cirq.num_qubits(lt_encoding.circuit)
        qubits = cirq.LineQubit.range(num_qubits)
        # initialize operator
        operation = lt_encoding.on(*qubits)

        qasm = openqasm(operation,rotation_allowed=True)
        assert qasm is not None
        for line in qasm:
            pass

    def test_LinearT_resources(self, lt_encoding):
        '''
        Tests estimate_resources executes without error.
        '''
        resources = estimate_resources(lt_encoding.circuit)