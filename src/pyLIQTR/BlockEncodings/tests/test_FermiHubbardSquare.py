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
from pyLIQTR.scheduler.scheduler import schedule_circuit
from pyLIQTR.scheduler.scheduler_utils import architecture, state_factory
from pyLIQTR.utils.circuit_decomposition import generator_decompose
import pyLIQTR.scheduler.sets as sets
from qualtran._infra.bloq import DecomposeNotImplementedError
from pyLIQTR.utils.global_ancilla_manager import gam


class TestFermiHubbardSquareEncoding:

    @pytest.fixture(scope="class")
    def fh_instance(self):
        model = getInstance('FermiHubbard',shape=(4,4), J=-1.0, U=4.0,cell=SquareLattice)
        return model

    @pytest.mark.parametrize("N",[3,4,7,8])
    def test_FermiHubbardSquare_decomposes_cirq_SQM(self, N):
        '''
        Tests gate decomposition existence.
        '''
        model  =  getInstance('FermiHubbard',shape=(N,N), J=-1, U=4, cell=SquareLattice)
        encoding = getEncoding(instance=model, encoding=VALID_ENCODINGS.FermiHubbardSquare)
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
        for op in decomposed:
            if 'ModAddK' in str(op):
                raise DecomposeNotImplementedError
            
    @pytest.mark.parametrize("N",[3,4,7,8])
    def test_FermiHubbardSquare_decomposes_generator_gam(self, N):
        '''
        Tests gate decomposition with global ancilla manager.
        '''
        model  =  getInstance('FermiHubbard',shape=(N,N), J=-1, U=4, cell=SquareLattice)
        encoding = getEncoding(instance=model, encoding=VALID_ENCODINGS.FermiHubbardSquare)
        decomposed = generator_decompose(encoding.circuit,context=cirq.DecompositionContext(gam))
        for op in decomposed:
            if 'ModAddK' in str(op):
                raise DecomposeNotImplementedError
            
    @pytest.mark.parametrize("N",[3,4,7,8])
    def test_FermiHubbardSquare_decomposes_generator_sqm(self, N):
        '''
        Tests gate decomposition with simple qubit manager.
        '''
        model  =  getInstance('FermiHubbard',shape=(N,N), J=-1, U=4, cell=SquareLattice)
        encoding = getEncoding(instance=model, encoding=VALID_ENCODINGS.FermiHubbardSquare)
        decomposed = generator_decompose(encoding.circuit,context=cirq.DecompositionContext(cirq.SimpleQubitManager()))
        for op in decomposed:
            if 'ModAddK' in str(op):
                raise DecomposeNotImplementedError

    @pytest.mark.parametrize("N",[3,4,7,8])
    def test_FermiHubbardSquare_qasm_sqm(self,N):
        '''
        Tests qasm printing functionality with simple qubit manager.
        '''
        # create registers
        model  =  getInstance('FermiHubbard',shape=(N,N), J=-1, U=4, cell=SquareLattice)
        encoding = getEncoding(instance=model, encoding=VALID_ENCODINGS.FermiHubbardSquare)
        num_qubits = cirq.num_qubits(encoding.circuit)
        qubits = cirq.LineQubit.range(num_qubits)
        # initialize operator
        operation = encoding.on(*qubits)

        qasm = openqasm(operation,rotation_allowed=True, context=cirq.DecompositionContext(cirq.SimpleQubitManager()))
        assert qasm is not None
        for line in qasm:
            pass

    @pytest.mark.parametrize("N",[3,4,7,8])
    def test_FermiHubbardSquare_qasm_gam(self,N):
        '''
        Tests qasm printing functionality with global ancilla manager.
        '''
        # create registers
        model  =  getInstance('FermiHubbard',shape=(N,N), J=-1, U=4, cell=SquareLattice)
        encoding = getEncoding(instance=model, encoding=VALID_ENCODINGS.FermiHubbardSquare)
        num_qubits = cirq.num_qubits(encoding.circuit)
        qubits = cirq.LineQubit.range(num_qubits)
        # initialize operator
        operation = encoding.on(*qubits)

        qasm = openqasm(operation,rotation_allowed=True, context=cirq.DecompositionContext(gam))
        assert qasm is not None
        for line in qasm:
            pass

    def test_FermiHubbardSquare_resources(self, fh_instance):
        '''
        Tests estimate_resources executes without error.
        '''
        encoding = getEncoding(instance=fh_instance, encoding=VALID_ENCODINGS.FermiHubbardSquare)
        resources = estimate_resources(encoding.circuit)

    @pytest.mark.parametrize("N",[3,4,7,8])
    def test_FermiHubbardSquare_scheduler_custom_gateset_sqm(self,N):
        '''
        Tests scheduler custom gateset functionality with simple qubit manager.
        '''
        model  =  getInstance('FermiHubbard',shape=(N,N), J=-1, U=4, cell=SquareLattice)
        encoding = getEncoding(instance=model, encoding=VALID_ENCODINGS.FermiHubbardSquare)
        my_timings = {sets.T: 10, sets.CLIFFORD: 1, sets.MISC: 0}
        my_params = architecture(user_input_timings=my_timings)
        res = schedule_circuit(encoding.circuit, full_profile=True, decomp_level=0, context=cirq.DecompositionContext(cirq.SimpleQubitManager()),
                               architecture_config=my_params)

    @pytest.mark.parametrize("N",[3,4,7,8])
    def test_FermiHubbardSquare_scheduler_custom_gateset_gam(self,N):
        '''
        Tests scheduler custom gateset functionality with global ancilla manager.
        '''
        model  =  getInstance('FermiHubbard',shape=(N,N), J=-1, U=4, cell=SquareLattice)
        encoding = getEncoding(instance=model, encoding=VALID_ENCODINGS.FermiHubbardSquare)
        my_timings = {sets.T: 10, sets.CLIFFORD: 1, sets.MISC: 0}
        my_params = architecture(user_input_timings=my_timings)
        res = schedule_circuit(encoding.circuit, full_profile=True, decomp_level=0,context=cirq.DecompositionContext(gam),
                               architecture_config=my_params)